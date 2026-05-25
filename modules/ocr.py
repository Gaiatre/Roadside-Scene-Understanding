import os
import time
import json
from PIL import Image
from chandra.model.hf import load_model, generate_hf
from chandra.model.schema import BatchInputItem


# =========================
# OCR GATING FUNCTION
# =========================
def should_run_ocr(frame_data):
    description = frame_data.get("description", "").lower()
    signs = frame_data.get("signs", [])

    keywords = [
        "shop", "sign", "board", "text",
        "banner", "advertisement", "store",
        "name", "building", "pharma", "hotel"
    ]

    # If explicit signs detected → run OCR
    if signs and len(signs) > 0:
        return True

    # If description hints at text presence → run OCR
    if any(k in description for k in keywords):
        return True

    return False


# =========================
# MAIN OCR FUNCTION
# =========================
def run_ocr(frames_dir, output_root, frame_caption_json=None):

    ocr_dir = output_root
    os.makedirs(ocr_dir, exist_ok=True)

    print("Loading Chandra model...")
    model = load_model()

    # -----------------------------
    # LOAD FRAME CAPTIONS (OPTIONAL)
    # -----------------------------
    frame_data_map = {}

    if frame_caption_json and os.path.exists(frame_caption_json):
        print("Loading frame captions for OCR gating...")

        with open(frame_caption_json, "r") as f:
            frame_json = json.load(f)

        # -------- CASE 1: list of frames --------
        if isinstance(frame_json, list):
            for frame in frame_json:
                if isinstance(frame, dict) and "frame_name" in frame:
                    frame_data_map[frame["frame_name"]] = frame

        # -------- CASE 2: dict --------
        elif isinstance(frame_json, dict):

            # Case 2A: {"frames": [...]}
            if "frames" in frame_json:
                for frame in frame_json["frames"]:
                    if isinstance(frame, dict) and "frame_name" in frame:
                        frame_data_map[frame["frame_name"]] = frame

            # Case 2B: {"frame_00001": {...}}
            else:
                frame_data_map = frame_json

    else:
        print("No frame captions provided → running OCR on all frames")

    # -----------------------------
    # GET IMAGE FILES
    # -----------------------------
    image_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith(".jpg")
    ])

    print(f"Found {len(image_files)} frames")

    skipped = 0

    for idx, image_name in enumerate(image_files, 1):

        base = os.path.splitext(image_name)[0]
        image_path = os.path.join(frames_dir, image_name)

        print(f"\n[{idx}/{len(image_files)}] Checking {image_name}")

        # -----------------------------
        # OCR GATING CHECK
        # -----------------------------
        if base in frame_data_map:
            frame_data = frame_data_map[base]

            if not should_run_ocr(frame_data):
                print("⏭ Skipping OCR (no text likely)")
                skipped += 1
                continue

        print("✅ Running OCR...")

        img = Image.open(image_path).convert("RGB")

        batch = [
            BatchInputItem(
                image=img,
                prompt=None,
                prompt_type="ocr_layout"
            )
        ]

        start_time = time.time()
        results = generate_hf(batch, model)
        elapsed = time.time() - start_time

        raw_html = results[0].raw

        out_path = os.path.join(
            ocr_dir,
            f"{base}_ocr_layout.html"
        )

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(raw_html)

        print(f"Saved → {out_path}")
        print(f"Time taken: {elapsed:.2f} seconds")

    print(f"\nOCR complete. Skipped {skipped} frames.")
    return ocr_dir
