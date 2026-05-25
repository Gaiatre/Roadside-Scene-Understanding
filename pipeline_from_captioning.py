import os

# ========= PATHS =========

VIDEO_PATH = "/home/gayatri/scene_rec/data/GH030002_clip_004.mp4"
FRAMES_DIR = "/home/gayatri/scene_rec/outputs/GH030002_clip_004/frames"
OUTPUT_DIR = "/home/gayatri/scene_rec/outputs/GH030002_clip_004"

OCR_HTML_DIR = os.path.join(OUTPUT_DIR, "ocr")
OCR_JSON_DIR = os.path.join(OUTPUT_DIR, "ocr_parsed")

os.makedirs(OCR_HTML_DIR, exist_ok=True)
os.makedirs(OCR_JSON_DIR, exist_ok=True)

print("\n===== STARTING FROM FRAME CAPTIONING =====\n")

# ============================================
# 1️⃣ FRAME CAPTIONING
# ============================================

from modules.caption_frames import caption_frames

frame_caption_path = os.path.join(OUTPUT_DIR, "frame_captions.json")

if os.path.exists(frame_caption_path):
    print("Frame captions already exist. Skipping frame captioning.")
else:
    frame_caption_path = caption_frames(
        frame_dir=FRAMES_DIR,
        output_dir=OUTPUT_DIR
    )
    print("Frame captions saved:", frame_caption_path)


# ============================================
# 2️⃣ VIDEO CAPTIONING
# ============================================

from modules.caption_video import caption_video

video_caption_path = os.path.join(OUTPUT_DIR, "video_caption.json")

if os.path.exists(video_caption_path):
    print("Video caption already exists. Skipping video captioning.")
else:
    video_caption_path = caption_video(
        video_path=VIDEO_PATH,
        output_root=OUTPUT_DIR
    )
    print("Video caption saved:", video_caption_path)


# ============================================
# 3️⃣ OCR (CPU Chandra)
# ============================================

from modules.ocr import run_ocr

# Check if OCR folder already has files
if os.path.exists(OCR_HTML_DIR) and len(os.listdir(OCR_HTML_DIR)) > 0:
    print("OCR outputs already exist. Skipping OCR.")
else:
    run_ocr(
        frames_dir=FRAMES_DIR,
        output_root=OCR_HTML_DIR,
        frame_caption_json=frame_caption_path
    )
# ============================================
# 4️⃣ OCR PARSING
# ============================================

from modules.ocr_parse import parse_ocr_folder

parse_ocr_folder(
    ocr_html_dir=OCR_HTML_DIR,
    output_root=OCR_JSON_DIR
)


# ============================================
# 5️⃣ FRAME + OCR MERGE
# ============================================

from modules.frame_fusion import merge_frames_with_ocr

frames_with_ocr_path = merge_frames_with_ocr(
    frame_caption_json_path=frame_caption_path,
    parsed_ocr_dir=OCR_JSON_DIR,
    output_root=OUTPUT_DIR
)


# ============================================
# 6️⃣ VIDEO + FRAME MERGE
# ============================================

from modules.video_fusion import merge_video_and_frames

fusion_input_path = merge_video_and_frames(
    frames_with_ocr_path=frames_with_ocr_path,
    video_json_path=video_caption_path,
    output_root=OUTPUT_DIR
)

# ============================================
# ENTITY MERGING
# ============================================

from modules.entity_merge import merge_entities

scene_with_entities_path = merge_entities(
    fusion_input_path=fusion_input_path,
    output_root=OUTPUT_DIR
)

# ============================================
# 7️⃣ FINAL LLM
# ============================================

# ============================================
# 6️⃣ FINAL LLM (Hierarchical reasoning)
# ============================================

from modules.final_llm import run_final_llm

run_final_llm(
    video_caption_path=video_caption_path,
    frames_ocr_fusion_path=scene_with_entities_path,
    output_root=OUTPUT_DIR
)
print("\n===== PIPELINE COMPLETE =====\n")
