import google.generativeai as genai
import json
import time
import os

# -------------------------
# 🔧 CONFIG (EDIT THESE)
# -------------------------
SCENE_JSON_PATH = "/home/gayatri/scene_rec/outputs/GH012788_clip_010/final_scene_representation.json"
OUTPUT_JSON_PATH = "/home/gayatri/scene_rec/outputs/GH012788_clip_010/vqa_output_counting.json"

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
Generate EXACTLY 10 question-answer pairs covering the question types below.
Questions may naturally overlap multiple types — this is encouraged, not penalized.

---

QUESTION TYPES TO COVER:

1. ENTITY AGGREGATION
   - Counting: exact integer counts of people, vehicles, shops, objects
   - Frequency: how often something appears (e.g., how many shops sell food-related goods)
   - Accumulation: running totals as the vehicle moves (e.g., total people seen by the time the vehicle passes X)
   - Summarization: aggregate properties across multiple entities (e.g., how many shops have shutters closed)

   - ALL answers must be exact integers
   - Banned words in answers: "many", "several", "some", "multiple", "a few"

2. TEMPORAL REASONING
   - Text appearance & disappearance: when does a shop/sign first appear or leave the frame
   - Sequencing order: what is the order establishments are encountered as the vehicle moves
   - Co-occurrence: which establishments are visible in the frame at the same time
   - Before/after landmark: what is seen before or after a specific shop or event

   - Answers must reference timestamps (e.g., "0:08") or ordinal position (e.g., "3rd") where relevant

3. SPATIAL GROUNDING
   - Relative position: left/right/in front of/behind a named anchor
   - Ordinal sequencing: which shop is Nth after a landmark
   - Multi-hop: chain two or more spatial relationships in one question
   - Cluster: which shops form a commercial group or block
   
   - Answers must use precise spatial primitives — never "nearby" or "close to"

4. LANGUAGE & PLACE OF ORIGIN DETECTION
   - Identify the language(s) visible on signage (e.g., Tamil, English, Hindi)
   - Infer regional or cultural origin from shop names, goods, or signage style
   - Identify script types visible (e.g., Latin, Devanagari, Tamil script)
   - Cross-reference multiple signs to infer the likely city/region/state
   - Answers should be specific (e.g., "Tamil", "South Indian", "Kerala") not vague ("Asian")

5. CROSS-TYPE REASONING (overlap encouraged)
   - Questions that require combining 2+ of the above types to answer
   - Examples:
     * "How many food-related shops appear before the temple entrance?" (aggregation + temporal + spatial)
     * "Which Tamil-script sign appears immediately after KMS TRADERS?" (language + spatial + temporal)
     * "By the time the vehicle passes UMA GOLD, how many people have been seen in total?" (temporal + aggregation)
6. OTHER 

WHAT THIS TESTS:
Question types that emerge naturally from the scene that do not fit cleanly 
into the above categories. These should reflect the core goal of this dataset:
understanding roadside scenes by combining visual cues AND textual information 
from signage, storefronts, advertisements, and public notices — in service of 
navigation, scene interpretation, and autonomous driving.

REQUIREMENTS:
- Each question must be genuinely different in reasoning type from the 5 categories above
- Each question must include a "inferred_category" field where you name and define 
  the question type you invented (e.g., "navigational_intent", "commercial_cluster_inference")
- The question must only be possible because BOTH visual scene details AND text/signage 
  are present — if it could be answered from visual cues alone or text alone, it does not qualify

THINK ABOUT:
- What would a navigation system need to know from this scene?
- What would an autonomous vehicle need to infer to make a decision?
- What semantic relationships between shops, signs, and events are non-obvious?
- What does the combination of text + visual context reveal that neither reveals alone?

EXAMPLES OF GOOD "OTHER" QUESTIONS:
- "Based on the sequence of shops visible, what type of neighborhood is this street 
   most likely serving?" (neighborhood_type_inference)
- "Which shop's signage provides the most actionable information for a vehicle 
   deciding whether to stop?" (navigational_relevance)
- "What does the co-location of a pharmacy, a temple, and a pilgrim store suggest 
   about the likely foot traffic pattern on this street?" (semantic_clustering)
- "If a delivery vehicle were looking for perishable goods vendors, how many 
   relevant stops exist on this street?" (task_oriented_navigation)

BANNED:
- Questions that are just counting, spatial, or temporal questions with a new label
- Questions answerable from visual cues alone (no text needed)
- Questions answerable from text alone (no scene context needed)
- Vague questions like "What can you infer about this street?" 

STRICT ENFORCEMENT:
- EVERY question in this category MUST have an "inferred_category" field
- If a question does not have an "inferred_category" field it does not count 
  towards the total — regenerate until you have exactly the number of questions specified with 
  this field
- Do NOT mix standard category questions into this block
- If you find yourself writing a counting, spatial, temporal, or language 
  question — stop and reframe it as a higher-order inference that uses 
  those as inputs but produces a new type of insight as output
---

STRICT RULES ACROSS ALL TYPES:
- NO yes/no questions
- NO questions answerable from a single sentence — prefer combining scene + temporal description
- NO repeating the same question structure more than 3 times
- MUST explicitly reference shop/landmark names from the description
- Answers must be concise: 1-15 words
- For integer answers: minimum value of 2 (avoid trivially guessable 0 or 1 answers)
- For ordering answers: use ordinal format ("1st", "2nd", "3rd")
- For spatial answers: always include a named anchor ("to the left of KMS TRADERS", not just "to the left")
- Include a "reasoning" field for any question requiring arithmetic or multi-hop inference:
  {{"question": "...", "answer": "...", "reasoning": "..."}}
  For all other questions the reasoning field can be omitted.

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

- Total = exactly 10

---
---

TEMPLATE-GUIDED QUESTION GENERATION (CLUSTER MODE):

Instead of generating arbitrary questions, you will be given a SMALL SET of related question templates (a template cluster).

Your job is to:

1. Understand the shared reasoning pattern across the templates
2. Determine which category (or combination of categories) they belong to
3. Generate multiple question-answer pairs that are INSPIRED by these templates

---

INPUT TEMPLATE CLUSTER:
VISUAL SIGNAGE GROUNDING
1. If a driver uses the sign featuring [VISUAL ELEMENT] as a landmark, what named establishment is [N] positions to its [left/right]?

2. What establishments are immediately to the left and right of the sign featuring [VISUAL ELEMENT]?

3. What is the position of the sign featuring [VISUAL ELEMENT] relative to [NAMED LANDMARK]?

4. What is the first named establishment visible after the sign featuring [VISUAL ELEMENT] appears in the [TIMESTAMP] window?

5. Among the establishments passed between [LANDMARK A] and [LANDMARK B], which one is distinguishable by a non-textual visual element on its signage, and what does that element indicate about its goods or services?

6. Between [LANDMARK A] and [LANDMARK B], which sign's visual element would most reliably help a driver confirm their position if all text on the signs were illegible?
---
VISUAL ELEMENT DEFINITION:
A visual element is ONLY:
- A human face or celebrity (e.g., actor Yash on a Pepsi ad)
- A mascot or illustrated character (e.g., the KFC colonel, a cartoon figure)
- An icon or symbol (e.g., a red cross, a crescent moon, a lotus)
- An animal or creature used as a brand symbol

NOT a visual element:
- Brand names or logos that are primarily text (Airtel, Coca-Cola banner)
- Building colours or architectural features
- Generic objects (a banner, a board, a sign)

If the scene description does not mention enough visual elements of this type,
skip templates that cannot be filled meaningfully — do not substitute with 
text-based brand identifiers.

TEMPLATE GUIDELINES:

- You MUST follow the SAME underlying reasoning structure as the templates
- You MAY vary wording, phrasing, and sentence structure naturally
- You MAY interpolate between templates if they share the same reasoning type
- You MUST NOT introduce entirely new reasoning styles outside this cluster BUT you can take creative freedom in similar type of questions/reasoning
-Every question must mention a different landmark
---

CONTROLLED FLEXIBILITY:

- Questions do NOT need to exactly match any single template
- But each question MUST clearly resemble at least ONE template in structure
- Minor variations in phrasing are encouraged to improve diversity
- The core reasoning pattern MUST remain consistent

---

CATEGORY INFERENCE:

For EACH generated QA pair, include:
"inferred_category": "<category_name>"

Where category_name is one of:
- "entity_aggregation"
- "temporal_reasoning"
- "spatial_grounding"
- "language_detection"
- "cross_type_reasoning"
- OR a new category if needed (define it briefly)

---

DISTRIBUTION REQUIREMENT:

- Generate EXACTLY 10 QA pairs
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
            if len(qa_pairs) != 10:
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


