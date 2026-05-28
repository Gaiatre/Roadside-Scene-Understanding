# Roadside Scene Understanding via Scene Text Recognition & VQA

Automated pipeline for generating semantically grounded Visual Question Answering (VQA) datasets from roadside driving videos in Indian multilingual environments. Built as part of an internship project at CVIT, IIIT Hyderabad.

---

## Overview

The pipeline takes GoPro-recorded roadside video clips, extracts frames, runs OCR to detect scene text, generates narrative scene descriptions using Gemini, and produces structured QA pairs across multiple reasoning categories — spatial, temporal, counting, language detection, navigational, and street character inference.

---

## Repository Structure

```
├── modules/                        # Core pipeline modules (OCR, captioning, description gen)
├── outputs/                        # Generated outputs (descriptions, QA JSONs, frames)
├── counting_qa.py                  # QA generation for all template categories
├── counting_qa2.py                 # QA generation for street character & activity
├── pipeline_from_captioning.py     # Full pipeline from frame extraction to scene description
├── qa_llama.py                     # QA generation using LLaMA (baseline)
├── video_grounding.py              # OCR bounding box grounding and coordinate validation
├── requirements.txt                # Python dependencies
└── .gitignore
```

---

## Modules (`modules/`)

These are the individual pipeline components called by `pipeline_from_captioning.py`. Each can also be run standalone with `python <module>.py`.

| File | Description |
|---|---|
| `frames.py` | Frame extraction — samples 1 frame every 2 seconds from the input clip using OpenCV |
| `caption_frames.py` | Frame-level captioning — generates a caption for each extracted frame via Gemini |
| `caption_video.py` | Video-level captioning — generates a single high-level caption for the full clip |
| `ocr.py` | Runs Chandra OCR on each frame to detect scene text and return raw bounding boxes |
| `ocr_parse.py` | Parses raw OCR bounding boxes into human-readable positional format (e.g. top-left, middle-right, bottom-right) for downstream use |
| `frame_fusion.py` | Merges OCR output with frame-level captions into a unified per-frame representation |
| `video_fusion.py` | Merges video-level captions with the already-fused frame+OCR data |
| `entity_merge.py` | Assigns unique entity IDs (E001, E002, ...) to each named entity in the description |
| `final_llm.py` | Passes the fully merged input (captions + OCR) through Gemini to produce the final narrative scene description |
| `clip.py` | Clips videos to ~1 minute segments — **must be run separately before the pipeline**, not called by `pipeline_from_captioning.py` |

**Note:** `clip.py` is a pre-processing step. Run it first on your raw GoPro footage before passing clips to the main pipeline:
```bash
python modules/clip.py --input raw_video.mp4 --output clips/ --duration 60
```

---

## File Descriptions

### `pipeline_from_captioning.py`
The main pipeline script. Runs end-to-end from a raw video clip through to a final scene description. Steps in order:
1. Extracts frames at 1 frame per 2 seconds using OpenCV
2. Generates frame-level captions via Gemini
3. Aggregates into a video-level caption
4. Merges with Chandra OCR output (detected text + bounding boxes)
5. Produces a final narrative scene description with entity IDs and temporal structure

**Usage:**
```bash
python pipeline_from_captioning.py 
```

---

### `counting_qa.py`
Generates QA pairs across all template clusters using the scene description as input. Covers:
- Spatial & ordinal navigation
- Entity aggregation & semantic filtering
- Temporal sequencing
- Navigational reasoning & hazard awareness
- Semantic & area inference
- Cultural & language detection

Uses template-guided prompting with controlled flexibility — the model follows template reasoning structure but varies phrasing to avoid rigid, formulaic outputs.

**Usage:**
```bash
python counting_qa.py
```

---

### `counting_qa2.py`
Generates QA pairs specifically for street character and activity inference. Questions probe street type (high-street, market strip, residential lane), activity zones, pedestrian intent, and commercial clustering. Extends the base QA generation with categories that require whole-street reasoning rather than entity-level lookup.

**Usage:**
```bash
python counting_qa2.py 
```

---

### `video_grounding.py`
Validates OCR bounding box placement by grounding detected text regions against QA answers. Also applies the coordinate rescaling fix for Chandra OCR's internal image resizing — bounding boxes are rescaled from the OCR's internal coordinate space back to the original frame dimensions before being used downstream.

**Usage:**
```bash
python video_grounding.py 
```

---

### `qa_llama.py`
Baseline QA generation using LLaMA 3B/7B instead of Gemini. Used for comparison during model evaluation. Does not support structured JSON output enforcement or exact count constraints reliably — retained for benchmarking purposes.

**Usage:**
```bash
python qa_llama.py 
```

---

## Environment Setup

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
```

**API key:** Set your Gemini API key as an environment variable before running any pipeline script:
```bash
export GEMINI_API_KEY=your_key_here
```

**Chandra OCR:** Can be downloaded from the [Chandra OCR Github](https://github.com/datalab-to/chandra)


---

## Data Format

Each processed clip produces three output files in `outputs/`:

**`description.txt`** — Narrative scene description with entity IDs, e.g.:
```
At 00:00, the vehicle passes Komal Cafe [E001] on the left, 
displaying a Pepsi advertisement featuring Actor Yash...
```

**`ocr_output.json`** — OCR detections with rescaled bounding boxes:
```json
{
  "frame_0010": [
    { "text": "Asian Bakery", "bbox": [120, 45, 380, 90], "confidence": 0.94 }
  ]
}
```

**`qa_pairs.json`** — Generated QA pairs:
```json
[
  {
    "question": "How many distinct medical establishments are passed in the entire clip?",
    "answer": "6",
    "category": "entity_aggregation",
    "reasoning": "Rajesh Medical Hall, S.B. Medical Hall, Shah Dental Care, Apollo Pharmacy, Dental Care Hospital, Dr. Kale's Clinic",
    "entity_ids_referenced": ["E004", "E009", "E011", "E015", "E018", "E023"]
  }
]
```

---

## Notes

- Videos should be trimmed to ~1 minute clips before processing. Full-length videos (~17 min) exceed the model's effective context window for description generation.
- The coordinate rescaling fix in `video_grounding.py` is required when using Chandra OCR — without it, bounding boxes will be offset from their actual positions in the original frame.
- LLaMA baseline (`qa_llama.py`) will not reliably enforce exact integer answers or structured output — use Gemini (`counting_qa.py`) for production QA generation.
