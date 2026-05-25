import google.generativeai as genai
import json
import time
import os

# -------------------------
# 🔧 CONFIG (EDIT THESE)
# -------------------------
SCENE_JSON_PATH = "/home/gayatri/scene_rec/outputs/GH030002_clip_004/final_scene_representation.json"
OUTPUT_JSON_PATH = "/home/gayatri/scene_rec/outputs/GH030002_clip_004/vqa_output_counting.json"

MODEL_NAME = "gemini-3-flash-preview"

# -------------------------
# 🔑 API KEY (from env)
# -------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not set. Run: export GEMINI_API_KEY=your_key")

# -------------------------
# 🔌 MODEL SETUP
# -------------------------
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    generation_config=genai.GenerationConfig(
        temperature=0.2,
        response_mime_type="application/json"
    )
)

# -------------------------
# 🧠 PROMPT
# -------------------------
def build_counting_prompt(video_id, narrative, temporal):

    narrative = narrative[:3700]
    temporal = temporal[:1000]

    return  f"""
You are a JSON generator for a Visual Question Answering dataset for street-level autonomous driving.
You output ONLY valid JSON. No explanation.

VIDEO ID:
{video_id}

SCENE DESCRIPTION:
{narrative}

TEMPORAL DESCRIPTION:
{temporal}

IMPORTANT:
All answers MUST be derived ONLY from the scene description and temporal description.
Do NOT hallucinate any shop names, people, objects, or events not explicitly mentioned.

TASK:
Generate EXACTLY 12 question-answer pairs covering the question types below.
Questions may naturally overlap multiple types — this is encouraged, not penalized.

---
QUALITY BAR:
Before finalizing each question, ask: "Could a person answer this without actually 
understanding the scene — just by skimming for keywords?"

If yes → discard and rewrite.

A good question requires the reader to:
- Combine at least 2 pieces of information from different parts of the description
- Perform some inference, arithmetic, or ordering — not just locate a fact
- Know the scene well enough that a random guess would likely be wrong

If a question could appear in a dataset about ANY street scene without changing 
the wording — it is too generic. Every question must be answerable ONLY because 
of specific details unique to this scene.
---
ANTI-LEAKAGE RULE:
The question must NEVER contain words or phrases that directly reveal or strongly 
imply the answer.

SELF CHECK: Read the question alone, without the answer. If the answer feels 
obvious from the question wording itself — rewrite it.
---
ANSWER CONCISENESS RULE:
- Factual answers (names, numbers, places): maximum 5 words
- Inferential answers (category "other"): maximum 10 words — 
  force yourself to summarize, not describe
- If your answer exceeds these limits, the question is asking too much — 
  split it or reframe it to have a tighter answer

BAD:  "Essential daily needs including medicine, fresh produce, beverages, 
       and food processing"
GOOD: "Essential daily and agricultural needs"

BAD:  "A narrow alleyway containing several parked motorbikes and a white truck"
GOOD: "The narrow alleyway beside A.C. VASUDEVAN & SON"
---
DISTRIBUTION REQUIREMENT:
Before closing the JSON, verify:

- Total = exactly 12

---
---

TEMPLATE-GUIDED QUESTION GENERATION (CLUSTER MODE):

Instead of generating arbitrary questions, you will be given a SMALL SET of related question templates (a template cluster).

Your job is to:

1. Understand the shared reasoning pattern across the templates
2. Determine which category (or combination of categories) they belong to
3. Generate multiple question-answer pairs that are INSPIRED by these templates

---
---
STREET CHARACTER & ACTIVITY
Core idea: Understand what kind of street this is and what kind of 
activity happens on it — through synthesis of the full scene, 
not shop-by-shop labeling.
TASK:
Generate EXACTLY 12 questions using the templates below.
Each question must be answerable ONLY from the scene and temporal descriptions.

---

TAXONOMY:

PRIMARY TYPOLOGY (choose one per scene):
- Arterial/High-Street: wide road, multi-story, branded signage, standardized
- Neighborhood Commercial: narrow, mixed-use, dense parked vehicles, manual shutters
- Market/Bazaar: extremely dense, overhead wires, temporary stalls, chaotic signage
- Institutional/Government: compound walls, public notices, sparse shops, tree coverage

VISUAL ATTRIBUTES (India-specific):
- Signage Style: Backlit-Flex, Painted-Wall, Neon, Digital, Hand-painted
- Encroachment Level: Clear, Sidewalk-occupied, Road-spill
- Visual Complexity: High, Medium, Low
- Temporal State: Closed(shutters-down), Active, Crowded

---
TEMPLATES:

T1 — STREET TYPE:
"Based on [specific physical features visible], what kind of street is this?"
- Requires: combining road width, building types, vehicle density, 
  and signage style into a single characterization
- Answer: street type + max 3 named supporting details (max 10 words)
- Do NOT use formal classification labels in the question or answer
- Answer should feel natural, not taxonomic
  BAD:  "Arterial/High-Street; multi-story buildings"
  GOOD: "Busy commercial high street; dense signage, multi-story buildings, heavy traffic"

T2 — DOMINANT ACTIVITY:
"What is the primary activity taking place on this street, 
 and which specific combination of shops, people, and vehicles supports this?"
- Requires: synthesizing the full commercial and social mix 
  into one dominant activity description
- Answer: activity description + 2-3 named supporting entities (max 12 words)
- Must not be answerable from a single shop or sentence alone

T3 — WHO USES THIS STREET:
"Based on the establishments and landmarks visible, 
 what kind of people most likely use this street and for what purpose?"
- Requires: inferring the primary user demographic from the 
  commercial mix, cultural markers, and visible people
- Answer: demographic description + inferred purpose (max 10 words)
- Must reference at least 2 named entities as evidence

T4 — ACTIVITY ZONES:
"Does this street have distinct zones of different activity, 
 and if so where does the transition happen?"
- Requires: comparing shop types, density, and landmarks across 
  the temporal segments to identify a shift in street character
- Answer: zone A description + zone B description + transition point (max 12 words)
- Must reference a named landmark or timestamp as the transition point

T5 — PEAK ACTIVITY POINT:
"At which point along this street is human activity at its highest, 
 and what specific combination of factors causes this?"
- Requires: identifying the segment with the highest concentration 
  of people, open shops, and vehicles
- Answer: named location or timestamp + 2 specific factors (max 10 words)


---
TEMPLATE GUIDELINES:

- You MUST follow the SAME underlying reasoning structure as the templates
- You MAY vary wording, phrasing, and sentence structure naturally
- You MAY interpolate between templates if they share the same reasoning type
- You MUST NOT introduce entirely new reasoning styles outside this cluster BUT you can take creative freedom in similar type of questions/reasoning

---

CONTROLLED FLEXIBILITY:

- Questions do NOT need to exactly match any single template
- But each question MUST clearly resemble at least ONE template in structure
- Minor variations in phrasing are encouraged to improve diversity
- The core reasoning pattern MUST remain consistent

---

DISTRIBUTION REQUIREMENT:

- Generate EXACTLY 12 QA pairs
- Ensure coverage across the provided template cluster
- Do NOT overuse a single phrasing pattern more than 3 times
- Maintain diversity in entities and anchors used

---

QUALITY REQUIREMENTS:

Each generated question must:
- Use DIFFERENT entities, timestamps, or anchors
- Require non-trivial reasoning (not direct lookup)(VERY IMPORTANT)
- Combine information from multiple parts of the scene when possible

---

STRUCTURAL CONSISTENCY CHECK:

Before finalizing each question, verify:
- Does this follow the SAME reasoning pattern as the templates?
- Would this question fit naturally within the same template family?

If not → rewrite.

---

-----------------------------------
ENTITY ID GROUNDING (ANSWERS ONLY)
-----------------------------------

You are given grounded entity names in the scene description, each with a unique ID:
Example: "KOMAL CAFE [shop_001]"

RULES FOR ANSWERS:

1. If an answer includes any shop, sign, building, or named entity that has an ID:
   you MUST include the ID in the answer.

   Format:
   "<name> [entity_id]"

2. NEVER include entity IDs in the QUESTION — ONLY in the ANSWER.

3. The name and ID must match exactly as given in the scene description.

4. If multiple entities are mentioned in the answer, ALL must include their IDs.

5. If an entity does not have an ID, you may refer to it normally.

Examples:

Correct:
Answer: "ASIAN BAKERY [shop_003]"
Answer: "KOMAL CAFE [shop_001], ASIAN SHAKES & ROLL'S [shop_002]"

Incorrect:
Answer: "ASIAN BAKERY"
Answer: "KOMAL CAFE [shop_002]"  (wrong ID)

-----------------------------------
---
OUTPUT FORMAT:
{{
  "video_id": "{video_id}",
  "qa_pairs": [
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "...", "reasoning": "..."}}
  ]
}}
"""
# -------------------------
# ⚙️ GENERATION
# -------------------------
def generate_counting_qa(video_id, narrative, temporal, retries=1):

    prompt = build_counting_prompt(video_id, narrative, temporal)

    for attempt in range(retries + 1):
        try:
            response = model.generate_content(prompt)
            data = json.loads(response.text)

            # 🔥 Handle case where model returns a list instead of dict
            if isinstance(data, list):
                qa_pairs = data
                data = {
                    "video_id": video_id,
                    "category": "counting_reasoning",
                    "qa_pairs": qa_pairs
                }
            else:
                qa_pairs = data.get("qa_pairs", [])

            # ✅ Validate count (FIXED → 15)
            if len(qa_pairs) != 12:
                print(f"⚠️ {video_id}: wrong count ({len(qa_pairs)})")
                continue

            print(f"✅ {video_id}")
            return data

        except json.JSONDecodeError:
            print(f"❌ {video_id} JSON parse error")
            time.sleep(1)

        except Exception as e:
            print(f"❌ {video_id} error: {e}")
            time.sleep(1)

    return {
        "video_id": video_id,
        "category": "counting_reasoning",
        "qa_pairs": [],
        "error": "generation_failed"
    }

# -------------------------
# 📂 LOAD DATA
# -------------------------
with open(SCENE_JSON_PATH, "r") as f:
    scenes = json.load(f)

if isinstance(scenes, dict):
    scenes = [scenes]

# -------------------------
# 🚀 RUN
# -------------------------
results = []

for i, scene in enumerate(scenes):
    video_id  = scene["video_id"]
    narrative = scene["narrative_scene_description"]
    temporal  = scene["temporal_progression_description"]

    print(f"[{i+1}/{len(scenes)}] Processing {video_id}")

    result = generate_counting_qa(video_id, narrative, temporal)
    results.append(result)

# -------------------------
# 💾 SAVE
# -------------------------
with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n🎉 Saved to {OUTPUT_JSON_PATH}")



