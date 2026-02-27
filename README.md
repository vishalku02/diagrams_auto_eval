# Coteach Eval: Pedagogical Quality of AI-Generated Math Diagrams

LLM-as-judge harness for evaluating whether VLMs can assess pedagogical quality of math diagrams using Mayer's CTML principles.

## Quick Start

```bash
pip install -r requirements.txt

# Run with a single model
python judge.py --csv data/sample.csv --models claude-sonnet-4-6 --limit 10

# Run with multiple models
python judge.py --csv data/sample.csv \
    --models claude-sonnet-4-6 gpt-5.2 gemini-3.0-pro \
    --concurrency 4

# Include an open-source model via Together API
python judge.py --csv data/sample.csv \
    --models claude-sonnet-4-6 meta-llama/Llama-4-Scout-17B-16E \
    --concurrency 4
```

## Environment Variables

```bash
export OPENAI_API_KEY="..."      # For GPT models
export ANTHROPIC_API_KEY="..."   # For Claude models
export GOOGLE_API_KEY="..."      # For Gemini models
export TOGETHER_API_KEY="..."    # For open-source models
```

## Structure

```
coteach_eval/
├── judge.py                     # Main eval harness
├── prompts/
│   └── pedagogical_quality.md   # CTML rubric prompt (workshop this!)
├── utils/
│   ├── models.py                # Multi-provider model factory
│   └── schema.py                # Pydantic output schema (5 CTML dims)
├── results/                     # Output CSVs
│   └── cache/                   # Per-diagram JSON cache (LRU)
└── requirements.txt
```

## 5 CTML Dimensions

| Dimension | Question |
|-----------|----------|
| Coherence | No extraneous visual elements? |
| Signaling | Visual cues guide attention? |
| Spatial Contiguity | Labels near their referents? |
| Segmenting | Complex info chunked? |
| Appropriate Labeling | Not over/under-labeled? |
