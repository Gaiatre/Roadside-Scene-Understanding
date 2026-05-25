import os
import json
import google.generativeai as genai

def caption_frames(frame_dir, output_dir, model_name="gemini-3-flash-preview"):

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)

    output_path = os.path.join(output_dir, "frame_captions.json")

    frame_paths = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.lower().endswith(".jpg")
    ])

    if not frame_paths:
        raise RuntimeError("No frames found in directory.")

    print(f"Processing {len(frame_paths)} frames...")

    model = genai.GenerativeModel(model_name)

    prompt = """
You are analyzing sequential driving frames.

For EACH image provided:
1. Identify the frame number.
2. Describe the scene in detail.
3. Mention the signs and what they say (exact transcription in english and the local language(like malayalam/odia/kannada))as well as identifiable people or characters on these signs.
4. Note vehicles, pedestrians, traffic signals, infrastructure.
5. Mention visible motion or scene changes across frames.
6. Mention any signs that are partially or fully occluded/obstructed and by what
Respond strictly in structured JSON format like:

{
  "frames": [
    {
      "frame_name": "...",
      "description": "..."
    }
  ]
}
"""

    contents = []

    for path in frame_paths:
        with open(path, "rb") as img_file:
            image_bytes = img_file.read()

        contents.append({
            "mime_type": "image/jpeg",
            "data": image_bytes
        })

    contents.append(prompt)

    print("Sending request to Gemini...")

    response = model.generate_content(contents)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    import re

    raw_text = response.text.strip()

    # Extract first JSON object from response
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)

    if not match:
        print("ERROR: No JSON object found in Gemini response")
        print(raw_text)
        raise ValueError("No JSON found")

    json_str = match.group(0)

    try:
        parsed = json.loads(json_str)
    except Exception as e:
        print("ERROR: Failed to parse extracted JSON")
        print(json_str)
        raise e

    import re

# Normalize frame_name to frame_00001 format
    for frame in parsed.get("frames", []):
        name = frame.get("frame_name", "")

        match = re.search(r"\d+", name)
        if match:
            num = int(match.group())
            frame["frame_name"] = f"frame_{num:05d}"
        else:
            print(f"Warning: Could not parse frame number from {name}")   
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)

    print("Saved combined frame captions to:", output_path)

    return output_path
