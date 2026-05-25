import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================================
# PATHS
# =====================================

scene_json_path = "outputs/GH012820_clip_014/final_scene_representation.json"
output_path = "outputs/GH012820_clip_014/generated_qa.json"

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# =====================================
# LOAD MODEL
# =====================================

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# =====================================
# LOAD SCENE DESCRIPTION
# =====================================

with open(scene_json_path, "r") as f:
    scene = json.load(f)

video_id = scene["video_id"]
narrative = scene.get("narrative_scene_description", "")
temporal = scene.get("temporal_progression_description", "")

# =====================================
# PROMPT
# =====================================

prompt = f"""
You are creating a dataset for autonomous driving and navigation AI.

You are given a roadside scene description extracted from a driving video.

Your job is to generate meaningful question-answer pairs for Visual Question Answering (VQA).

Scene Description:
{narrative}

Temporal Description:
{temporal}

Generate BETWEEN 5 AND 10 questions for EACH category below.

Categories:

1. text_detection
Questions about readable text such as shop names or signs.

2. spatial_grounding
Questions about spatial relations between places.

3. scene_text_integration
Questions where text influences interpretation of the scene.

4. counting_reasoning
Questions requiring counting objects or businesses.

5. navigation_reasoning
Questions about directions or path decisions.

6. commercial_understanding
Questions about types of establishments.

7. temporal_reasoning
Questions about order of appearance in the video.

Rules:
- Questions must be answerable using the description
- Avoid yes/no questions
- Answers must be concise
- Questions must sound natural
- Focus on navigation usefulness

Return STRICT JSON ONLY.

FORMAT:

{{
"video_id":"{video_id}",
"qa_pairs":{{
"text_detection":[{{"question":"","answer":""}}],
"spatial_grounding":[{{"question":"","answer":""}}],
"scene_text_integration":[{{"question":"","answer":""}}],
"counting_reasoning":[{{"question":"","answer":""}}],
"navigation_reasoning":[{{"question":"","answer":""}}],
"commercial_understanding":[{{"question":"","answer":""}}],
"temporal_reasoning":[{{"question":"","answer":""}}]
}}
}}
"""

# =====================================
# GENERATE
# =====================================

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

outputs = model.generate(
    **inputs,
    max_new_tokens=1200,
    temperature=0.8,
    top_p=0.9,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# =====================================
# EXTRACT JSON SAFELY
# =====================================

match = re.search(r"\{.*\}", response, re.DOTALL)

if match:
    json_str = match.group(0)

    try:
        qa_data = json.loads(json_str)
    except json.JSONDecodeError:
        qa_data = {
            "video_id": video_id,
            "error": "Invalid JSON returned",
            "raw_output": response
        }

else:
    qa_data = {
        "video_id": video_id,
        "error": "No JSON detected",
        "raw_output": response
    }

# =====================================
# SAVE OUTPUT
# =====================================

with open(output_path, "w") as f:
    json.dump(qa_data, f, indent=2)

print("QA dataset saved to:", output_path)
