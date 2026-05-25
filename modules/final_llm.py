import os
import json
import re
import google.generativeai as genai


def run_final_llm(
    video_caption_path,
    frames_ocr_fusion_path,
    output_root,
    model_name="gemini-3-flash-preview"
):

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)

    # -----------------------------
    # LOAD VIDEO CAPTION
    # -----------------------------
    with open(video_caption_path, "r", encoding="utf-8") as f:
        video_data = json.load(f)

    # -----------------------------
    # LOAD FRAME + OCR FUSION
    # -----------------------------
    with open(frames_ocr_fusion_path, "r", encoding="utf-8") as f:
        frame_data = json.load(f)

    video_summary = video_data
    frames = frame_data["frames"]
    entities = frame_data.get("entities", [])
    video_name = video_summary.get("video_id", "unknown_video")

    output_path = os.path.join(
        output_root,
        "final_scene_representation.json"
    )

    # -------------------------------------------------
    # PROMPT
    # -------------------------------------------------

    prompt = """
IMPORTANT: You MUST include entity IDs in the output as specified.
You are an expert multimodal perception analyst and spatial reasoning specialist.

You are given TWO levels of information:

-----------------------------------
LEVEL 1 — GLOBAL VIDEO CONTEXT
-----------------------------------

A video-level scene summary describing the overall environment.

-----------------------------------
LEVEL 2 — LOCAL FRAME OBSERVATIONS
-----------------------------------

Frame-by-frame observations extracted from the SAME video.

Each frame contains:
• scene descriptions
• detected shop names
• OCR text detections
• objects and vehicles
• pedestrians
• infrastructure

Frames are ordered sequentially in time.

Your task is to integrate BOTH levels of information to produce a navigation-friendly scene understanding.

IMPORTANT:
The viewer is assumed to be traveling forward along the same path as the recording vehicle.

-----------------------------------
SPATIAL NARRATION RULES
-----------------------------------

1. Describe scenes using RELATIVE POSITIONING between places.

Use phrases like:
• "To the left of [Place A] is [Place B]"
• "To the right of [Place B] is [Place C]"
• "[Place D] stands beside [Place C]"
• "Adjacent to [Place E] is [Place F]"

2. Build a continuous spatial chain.

Each sentence must connect a NEW place to a PREVIOUSLY mentioned place.

3. DO NOT use cinematic narration.

Avoid:
• moving forward
• ahead
• next
• after that
• then
• further along

4. Use shop names and readable signs naturally as landmarks.

5. Mention:

• shop names  
• signboards  
• visible businesses  
• pedestrians and their actions(make sure to do thisand mention the count if multiple present)  
• parked vehicles(if multiple mention the count)  
• alleyways  
• temples or landmarks  

6. Maintain geographic consistency.

7. Traffic and regulatory signs are critical navigation cues.

If visible, describe:

• STOP / YIELD / ONE WAY signs.  
• speed limits  
• parking rules  
• warning signs  

Include their **position relative to nearby landmarks**.

-----------------------------------
TEMPORAL DESCRIPTION
-----------------------------------

Also produce a second section:

temporal_progression_description

This must describe:

• the ORDER in which locations appear
• follow the ego vehicle's forward motion
• mention timestamp(very important)

Use phrases like:

• "At the beginning of the clip..."
• "Shortly after..."
• "As the vehicle continues forward..."
• "Further along the road..."
• "Near the end of the clip..."

Do NOT describe adjacency here.
-----------------------------------
GROUNDING WITH ENTITY IDS (STRICT)
-----------------------------------

You are also given a list of unified scene entities, each with:
• a unique ID (e.g., entity_001)
• a text label (e.g., "Airtel")

STRICT REQUIREMENTS:

1. EVERY shop, signboard, or named establishment that appears in the entity list
   MUST be written in this format:

   "<name> [entity_id]"

2. It is NOT allowed to mention a shop name without its ID if that entity exists.

3. If an entity is mentioned without an ID, the output is considered incorrect.

4. Maintain natural readability, but ALWAYS append the ID.

5. ONLY use IDs from the provided entity list.
   DO NOT invent new IDs such as:
   - shop_XXX
   - hotel_XXX
   - building_XXX
   - sign_XXX

Example:
Correct:
"KOMAL CAFE [entity_012] is adjacent to ASIAN BAKERY [entity_015]"

Incorrect:
"KOMAL CAFE [shop_001]"
"KOMAL CAFE is adjacent to ASIAN BAKERY"

-----------------------------------
-----------------------------------
OUTPUT FORMAT
-----------------------------------

Return STRICT JSON:

{
"video_id": "...",
"narrative_scene_description": "...",
"temporal_progression_description": "..."
}

"""

    # -----------------------------
    # ADD VIDEO SUMMARY
    # -----------------------------
    prompt += "\n\nGLOBAL VIDEO SUMMARY:\n"
    prompt += json.dumps(video_summary, indent=2, ensure_ascii=False)

    # -----------------------------
    # ADD FRAME DATA
    # -----------------------------
    prompt += "\n\nFRAME + OCR OBSERVATIONS:\n"
    prompt += json.dumps(frames, indent=2, ensure_ascii=False)

    print("Sending hierarchical scene reasoning request...")
    
    prompt += "\n\nUNIFIED ENTITY LIST (USE THESE IDS ONLY):\n"
    prompt += json.dumps(entities, indent=2, ensure_ascii=False)
    # -----------------------------
    # RUN MODEL
    # -----------------------------
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)

    raw_text = response.text.strip()

    # -----------------------------
    # EXTRACT JSON SAFELY
    # -----------------------------
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)

    if match:
        json_str = match.group(0)
        try:
            structured_output = json.loads(json_str)
        except json.JSONDecodeError:
            structured_output = {
                "video_id": video_name,
                "error": "Invalid JSON returned",
                "raw_output": raw_text
            }
    else:
        structured_output = {
            "video_id": video_name,
            "error": "No JSON found in model output",
            "raw_output": raw_text
        }

    # -----------------------------
    # SAVE OUTPUT
    # -----------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured_output, f, indent=2, ensure_ascii=False)

    print("Final structured JSON saved to:", output_path)

    return structured_output
