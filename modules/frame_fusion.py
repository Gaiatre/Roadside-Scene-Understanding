import os
import json


def merge_frames_with_ocr(frame_caption_json_path, parsed_ocr_dir, output_root):

    # Load frame caption JSON
    with open(frame_caption_json_path, "r", encoding="utf-8") as f:
        frame_data = json.load(f)

    output_path = os.path.join(output_root, "frames_with_ocr.json")

    # Iterate over frames
    for frame in frame_data["frames"]:

        # Example frame_name: "frame_1"
        frame_name = frame["frame_name"]

        # Extract number from "frame_1"
        number_part = int(frame_name.split("_")[1])

        # Convert to zero-based index and match OCR naming
        # frame_1 → frame_00000
        base = f"frame_{number_part - 1:05d}"

        ocr_filename = f"{base}_ocr.json"
        ocr_path = os.path.join(parsed_ocr_dir, ocr_filename)

        if os.path.exists(ocr_path):
            with open(ocr_path, "r", encoding="utf-8") as f:
                ocr_data = json.load(f)
        else:
            ocr_data = []

        frame["ocr_detections"] = ocr_data

    # Save merged output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(frame_data, f, indent=2, ensure_ascii=False)

    print("Saved merged frames+OCR:", output_path)

    return output_path
