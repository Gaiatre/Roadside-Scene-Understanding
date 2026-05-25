import os
import json
import google.generativeai as genai


def caption_video(video_path, output_root, model_name="gemini-3-flash-preview"):

    # =========================
    # API KEY
    # =========================
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)

    # =========================
    # Paths
    # =========================
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_root, "video_caption.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # =========================
    # Load Video Bytes
    # =========================
    print(f"Reading video file: {video_path}")

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    model = genai.GenerativeModel(model_name)

    prompt = """
Analyze this driving footage and provide:

1. Scene Setting
2. Vehicle Analysis
3. Traffic Dynamics
4. Temporal Progression
5. Safety Observations
6. Environmental Context
7. Mention any alleyways or pathways that are visible.
Respond in structured text.
"""

    print("Sending video to Gemini...")

    response = model.generate_content([
        {
            "mime_type": "video/mp4",
            "data": video_bytes
        },
        prompt
    ])

    result = {
        "video_id": video_name,
        "model_caption": response.text
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print("Saved video caption to:", output_path)

    return output_path
