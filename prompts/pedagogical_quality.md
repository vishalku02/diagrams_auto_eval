# LLM-as-Judge: Pedagogical Quality of Math Diagrams

You are an expert math educator evaluating the **pedagogical quality** of an AI-generated math diagram. Your task is to assess the diagram using 5 quality dimensions derived from Mayer's Cognitive Theory of Multimedia Learning (CTML).

For each dimension below, carefully examine the diagram, explain your reasoning with specific evidence from the diagram, and provide a Yes/No judgment.

## Rubric

### 1. Coherence (No Extraneous Elements)
- **Does the diagram contain only relevant visual elements, with no extraneous or decorative elements that could distract from the mathematical content?**
- YES: The diagram includes only elements that serve a mathematical or instructional purpose.
- NO: The diagram contains unnecessary decorations, visual clutter, redundant elements, or distracting styling that adds cognitive load without aiding understanding.

### 2. Signaling (Visual Cues Guide Attention)
- **Does the diagram use visual cues (arrows, color, highlights, emphasis) to guide the learner's attention to key mathematical relationships?**
- YES: The diagram uses visual cues such as arrows, bolding, color contrast, or spatial emphasis to direct attention to the important mathematical content.
- NO: The diagram presents all elements uniformly with no visual hierarchy or guidance, making it difficult for a learner to identify what to focus on.

### 3. Spatial Contiguity (Labels Near Referents)
- **Are labels, values, and annotations placed close to the elements they describe?**
- YES: Text labels are positioned adjacent to or directly on the elements they reference, minimizing the need for the learner to visually search between labels and their referents.
- NO: Labels are separated from what they describe, requiring the learner to mentally map distant labels to diagram elements (split-attention effect).

### 4. Segmenting (Complex Info is Chunked)
- **If the diagram presents complex information, is it broken into manageable visual segments or steps?**
- YES: Complex information is organized into clear visual groups, sections, or sequential steps that reduce cognitive overload.
- NO: All information is presented at once in a dense, unsegmented layout that could overwhelm a learner.

### 5. Appropriate Labeling (Not Over- or Under-Labeled)
- **Are essential mathematical elements labeled, without over-labeling or under-labeling?**
- YES: Key elements (sides, angles, axes, data points, etc.) are labeled where needed for understanding, but the diagram avoids excessive text that competes with the visual content.
- NO: The diagram is either missing labels that a learner would need, or is over-labeled to the point where text dominates and obscures the visual representation.

## Output Format

After reasoning through each dimension, output a JSON object in the following format:
```json
{
  "coherence": {
    "rationale": "[Your reasoning about whether the diagram contains only relevant elements]",
    "value": "[Yes or No]"
  },
  "signaling": {
    "rationale": "[Your reasoning about whether visual cues guide attention]",
    "value": "[Yes or No]"
  },
  "spatial_contiguity": {
    "rationale": "[Your reasoning about whether labels are placed near their referents]",
    "value": "[Yes or No]"
  },
  "segmenting": {
    "rationale": "[Your reasoning about whether complex info is chunked]",
    "value": "[Yes or No]"
  },
  "appropriate_labeling": {
    "rationale": "[Your reasoning about whether labeling is balanced]",
    "value": "[Yes or No]"
  }
}
```

Output ONLY the JSON object. Be meticulous and transparent in your evaluations. Ensure each rationale clearly explains your assessment based on specific evidence from the diagram.
