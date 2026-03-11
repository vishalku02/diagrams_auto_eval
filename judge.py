"""Run CTML pedagogical quality evaluations across multiple VLMs.

Usage:
    python judge.py --csv data/sample.csv --models claude-sonnet-4-6 gpt-5.2 gemini-3.0-pro --limit 10
    python judge.py --csv data/sample.csv --models claude-sonnet-4-6 --concurrency 4
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import ValidationError
from tqdm import tqdm

from utils.models import get_judge, detect_provider
from utils.schema import STRICT_RESPONSE_FORMAT, PedagogicalJudgeOutput


CACHE_ROOT = Path("results/cache")
RESULTS_ROOT = Path("results")
PROMPT_PATH = Path("prompts/pedagogical_quality.md")
RUBRIC_DIMENSIONS = ["coherence", "signaling", "label_accuracy", "labeling"]


def load_prompt() -> str:
    """Load the pedagogical quality prompt from markdown."""
    return PROMPT_PATH.read_text(encoding="utf-8")


def prompt_hash(prompt_template: str) -> str:
    return hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()[:12]


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_").replace(".", "-")


def _normalize_path(value: Any) -> Optional[Path]:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return Path(normalized)


def _normalize_string(value: Any) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_samples(data_path: Path, limit: int = 0) -> list[dict]:
    """Load diagram samples from CSV or JSON."""
    samples = []
    if data_path.suffix.lower() == ".json":
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"JSON input must be an array: {data_path}")
        for idx, row in enumerate(payload):
            if not isinstance(row, dict):
                continue
            diagram_id = _normalize_string(row.get("diagram_id")) or _normalize_string(row.get("idx")) or str(idx)
            image_path = _normalize_path(
                row.get("image_png_path")
                or row.get("image")
                or row.get("file-path")
                or row.get("file_path")
            )
            samples.append({
                "diagram_id": diagram_id,
                "image_path": image_path,
                "prompt_text": _normalize_string(row.get("prompt")),
            })
    else:
        df = pd.read_csv(data_path)
        for idx, row in df.iterrows():
            diagram_id = _normalize_string(row.get("diagram_id")) or str(idx)
            image_path = _normalize_path(row.get("image_png_path", row.get("image", "")))
            samples.append({
                "diagram_id": diagram_id,
                "image_path": image_path,
                "prompt_text": _normalize_string(row.get("prompt")),
            })

    if limit > 0:
        samples = samples[:limit]
        print(f"Limited to {len(samples)} samples.")
    else:
        print(f"Loaded {len(samples)} samples.")
    return samples


def build_evaluation_prompt(prompt_template: str, sample_prompt: str) -> str:
    if not sample_prompt:
        return prompt_template
    return (
        f"{prompt_template}\n\n"
        "## Diagram Prompt\n"
        f"{sample_prompt}\n"
    )


async def evaluate_model(
    model_name: str,
    samples: list[dict],
    prompt_template: str,
    prompt_cache_key: str,
    *,
    temperature: float = 0.0,
    concurrency: int = 4,
    max_retries: int = 3,
) -> list[dict]:
    """Run evaluations for a single model across all samples."""
    judge = get_judge(model_name)
    sem = asyncio.Semaphore(concurrency)
    model_dir = CACHE_ROOT / safe_model_name(model_name) / f"prompt_{prompt_cache_key}"
    model_dir.mkdir(parents=True, exist_ok=True)

    results = []
    provider = detect_provider(model_name)
    # Only OpenAI supports strict JSON response_format natively
    use_response_format = provider == "openai"

    async def evaluate_one(sample: dict) -> Optional[dict]:
        cache_path = model_dir / f"diagram_{sample['diagram_id']}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())

        async with sem:
            image_bytes = None
            if sample["image_path"] and Path(sample["image_path"]).exists():
                image_bytes = Path(sample["image_path"]).read_bytes()

            attempt = 0
            while attempt < max_retries:
                start = time.perf_counter()
                try:
                    evaluation_prompt = build_evaluation_prompt(prompt_template, sample["prompt_text"])
                    resp = await judge.call(
                        prompt=evaluation_prompt,
                        image_bytes=image_bytes,
                        temperature=temperature,
                        response_format=STRICT_RESPONSE_FORMAT if use_response_format else None,
                    )
                except Exception as e:
                    attempt += 1
                    if attempt >= max_retries:
                        print(f"  API error for diagram {sample['diagram_id']}: {e}")
                        return None
                    await asyncio.sleep(2 ** attempt)
                    continue

                elapsed_ms = (time.perf_counter() - start) * 1000.0

                # Try to parse structured output
                try:
                    # Extract JSON from response (may be wrapped in markdown)
                    text = resp.text.strip()
                    if text.startswith("```"):
                        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                    structured = PedagogicalJudgeOutput.model_validate_json(text)
                    break
                except (ValidationError, json.JSONDecodeError) as exc:
                    attempt += 1
                    if attempt >= max_retries:
                        print(f"  Parse error for diagram {sample['diagram_id']} ({model_name}): {exc}")
                        error_record = {
                            "diagram_id": sample["diagram_id"],
                            "model": model_name,
                            "prompt_hash": prompt_cache_key,
                            "error": str(exc),
                            "raw_response": resp.text[:500],
                        }
                        cache_path.write_text(json.dumps(error_record, indent=2))
                        return error_record
                    continue

            record = {
                "diagram_id": sample["diagram_id"],
                "model": model_name,
                "prompt_hash": prompt_cache_key,
                "elapsed_ms": elapsed_ms,
                "tokens": {
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cached_tokens": resp.cached_tokens,
                    "total_tokens": resp.total_tokens,
                },
                "rubric": structured.model_dump(),
            }
            cache_path.write_text(json.dumps(record, indent=2))
            return record

    tasks = [asyncio.create_task(evaluate_one(s)) for s in samples]
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"  {model_name}"):
        result = await fut
        if result:
            results.append(result)

    return results


def export_results(all_results: list[dict], output_root: Path, *, run_tag: str) -> tuple[Path, Path]:
    """Export per-run results to JSON and flattened CSV."""
    output_root.mkdir(parents=True, exist_ok=True)

    # Keep prompt hash internal for cache/versioning, but do not export it.
    export_records = [{k: v for k, v in record.items() if k != "prompt_hash"} for record in all_results]

    json_path = output_root / f"pedagogical_eval_{run_tag}.json"
    json_path.write_text(json.dumps(export_records, indent=2), encoding="utf-8")

    rows = []
    for record in export_records:
        rubric = record.get("rubric", {})
        tokens = record.get("tokens", {})
        row = {
            "diagram_id": record.get("diagram_id"),
            "model": record.get("model"),
            "elapsed_ms": record.get("elapsed_ms"),
            "input_tokens": tokens.get("input_tokens"),
            "output_tokens": tokens.get("output_tokens"),
            "cached_tokens": tokens.get("cached_tokens"),
            "error": record.get("error"),
        }
        # Flatten rubric dimensions
        for dim in RUBRIC_DIMENSIONS:
            dim_data = rubric.get(dim, {})
            row[f"{dim}_value"] = dim_data.get("value")
            row[f"{dim}_rationale"] = dim_data.get("rationale")
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output_root / f"pedagogical_eval_{run_tag}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nExported JSON results to {json_path}")
    print(f"Exported CSV results to {csv_path}")
    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Input CSV or JSON with diagram samples")
    parser.add_argument("--models", nargs="+", required=True, help="Model names to evaluate")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Cap on number of samples (0 = all)")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.csv}")

    prompt_template = load_prompt()
    prompt_cache_key = prompt_hash(prompt_template)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    samples = load_samples(args.csv, limit=args.limit)

    all_results = []
    for model_name in args.models:
        print(f"\nEvaluating with {model_name} ({detect_provider(model_name)})...")
        results = asyncio.run(evaluate_model(
            model_name,
            samples,
            prompt_template,
            prompt_cache_key,
            temperature=args.temperature,
            concurrency=args.concurrency,
            max_retries=args.max_retries,
        ))
        all_results.extend(results)

    export_results(all_results, RESULTS_ROOT, run_tag=run_tag)


if __name__ == "__main__":
    main()
