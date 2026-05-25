import os
import json
import re
import cv2


# -------------------------
# PATHS
# -------------------------
SCENE_PATH = "/home/gayatri/scene_rec/outputs/GH012788_clip_010/scene_with_entities.json"
QA_PATH = "/home/gayatri/scene_rec/outputs/GH012788_clip_010/vqa_output_counting.json"
FRAMES_DIR = "/home/gayatri/scene_rec/outputs/GH012788_clip_010/frames"
OUTPUT_DIR = "/home/gayatri/scene_rec/outputs/GH012788_clip_010/qa_videos"

FPS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------
# LOAD DATA
# -------------------------
with open(SCENE_PATH) as f:
    scene = json.load(f)

with open(QA_PATH) as f:
    qa_data = json.load(f)

if isinstance(qa_data, list):
    qa_data = qa_data[0]

qa_pairs = qa_data["qa_pairs"]
entities = {e["id"]: e for e in scene["entities"]}


# -------------------------
# HELPERS
# -------------------------
def extract_ids(text):
    return re.findall(r"\[(.*?)\]", text)


def frame_to_index(frame_id):
    return int(frame_id.split("_")[1]) - 1


# -------------------------
# BEST BBOX SELECTION
# -------------------------
def get_best_bbox(instances):
    """
    Pick the bbox closest to average vertical position
    (stable across frames)
    """
    if not instances:
        return None

    y_centers = [
        (inst["bbox"][1] + inst["bbox"][3]) / 2
        for inst in instances
    ]

    avg_y = sum(y_centers) / len(y_centers)

    return min(
        instances,
        key=lambda inst: abs(
            ((inst["bbox"][1] + inst["bbox"][3]) / 2) - avg_y
        )
    )


# -------------------------
# DRAW BBOX (FINAL CLEAN)
# -------------------------
def draw_bbox(image, bbox, label):
    h_img, w_img = image.shape[:2]

    x1, y1, x2, y2 = bbox
    if y2 > h_img:
        scale = h_img / 720.0   # 👈 fixed reference

        x1 = int(x1 * scale)
        x2 = int(x2 * scale)
        y1 = int(y1 * scale)
        y2 = int(y2 * scale)

    # -------------------------
    # 🔥 FIX: vertical correction
    # -------------------------

    print("\n--- BEFORE ---", bbox)
    # -------------------------
    # 🎯 CONDITIONAL SHIFT (your idea, but controlled)
    # -------------------------
    center_y = (y1 + y2) / 2
    height = y2 - y1

    if center_y > (h_img / 2):
        # lower half → stronger correction
        shift_up = int(height * 1.2)
    else:
        # upper half → mild/no correction
        shift_up = int(height * 0.2)

    y1 -= shift_up
    y2 -= shift_up
    print("--- AFTER ---", [x1, y1, x2, y2])
    # clip
    x1 = max(0, min(int(x1), w_img))
    x2 = max(0, min(int(x2), w_img))
    y1 = max(0, min(int(y1), h_img))
    y2 = max(0, min(int(y2), h_img))

    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, label, (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# -------------------------
# MAIN LOOP
# -------------------------
for i, qa in enumerate(qa_pairs):

    answer = qa["answer"]
    entity_ids = extract_ids(answer)

    print("\nENTITY CHECK:")
    for eid in entity_ids:
        print(eid, "→", entities.get(eid, {}).get("text"))

    if not entity_ids:
        continue

    print(f"\n🎯 QA {i+1}: {entity_ids}")

    # -------------------------
    # Collect relevant frames
    # -------------------------
    frame_indices = set()

    for eid in entity_ids:
        if eid not in entities:
            continue

        for inst in entities[eid]["instances"]:
            idx = frame_to_index(inst["frame_id"])
            frame_indices.add(idx)

    if not frame_indices:
        print("⚠️ No frames found")
        continue

    min_f = min(frame_indices)
    max_f = max(frame_indices)

    min_f = max(1, min_f - 2)
    max_f = max_f + 2

    print(f"Frames: {min_f} → {max_f}")

    # -------------------------
    # Video writer
    # -------------------------
    sample_frame = os.path.join(FRAMES_DIR, f"frame_{min_f:05d}.jpg")
    img = cv2.imread(sample_frame)

    if img is None:
        print("⚠️ Failed to read sample frame")
        continue

    h, w, _ = img.shape

    out_path = os.path.join(OUTPUT_DIR, f"qa_{i+1}.avi")

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        FPS,
        (w, h)
    )

    # -------------------------
    # Generate video
    # -------------------------
    for fidx in range(min_f, max_f + 1):

        frame_path = os.path.join(FRAMES_DIR, f"frame_{fidx:05d}.jpg")

        if not os.path.exists(frame_path):
            continue

        image = cv2.imread(frame_path)

        for eid in entity_ids:
            if eid not in entities:
                continue

        for inst in entities[eid]["instances"]:
            
            if frame_to_index(inst["frame_id"]) == fidx:
                label = f"{entities[eid]['text']} [{eid}]"
                draw_bbox(image,inst["bbox"], label)

        writer.write(image)

    writer.release()
    print(f"✅ Saved: {out_path}")


print("\n🎬 All QA videos generated!")
