import os
import json


def merge_video_and_frames(video_json_path, frames_with_ocr_path, output_root):

    with open(video_json_path, "r", encoding="utf-8") as f:
        video_data = json.load(f)

    with open(frames_with_ocr_path, "r", encoding="utf-8") as f:
        frame_data = json.load(f)

    video_name = video_data["video_id"]

    output_path = os.path.join(output_root,  "fusion_input.json")

    fusion_data = {
        "video_summary": video_data,
        "frames": frame_data["frames"]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fusion_data, f, indent=2, ensure_ascii=False)

    print("Saved fusion input:", output_path)

    return output_path
