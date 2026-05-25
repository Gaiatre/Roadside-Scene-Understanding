import os
import json
import re


# -------------------------
# TEXT CLEANING
# -------------------------
def normalize(text):
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"[^a-z0-9\s&']", "", text)  # keep useful chars
    text = " ".join(text.split())
    return text.strip()


def is_similar(text1, text2):
    """
    STRICT match after normalization
    """
    return text1 == text2


# -------------------------
# MAIN MERGE FUNCTION
# -------------------------
def merge_entities(fusion_input_path, output_root):

    with open(fusion_input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])

    # =========================
    # 1️⃣ Collect detections
    # =========================
    all_detections = []

    for frame in frames:
        frame_id = frame.get("frame_name")

        for det in frame.get("ocr_detections", []):
            raw_text = det.get("text", "").strip()

            if not raw_text:
                continue

            norm_text = normalize(raw_text)

            # 🚫 skip very short/noisy text
            if len(norm_text) < 3:
                continue

            all_detections.append({
                "frame_id": frame_id,
                "text": raw_text,
                "norm_text": norm_text,
                "bbox": det["bbox"],
                "local_id": det.get("local_id")
            })

    print(f"Total OCR detections collected: {len(all_detections)}")

    # =========================
    # 2️⃣ Cluster by EXACT norm_text
    # =========================
    cluster_map = {}

    for det in all_detections:
        key = det["norm_text"]

        if key not in cluster_map:
            cluster_map[key] = {
                "norm_text": key,
                "texts": [],
                "instances": []
            }

        cluster_map[key]["texts"].append(det["text"])
        cluster_map[key]["instances"].append(det)

    print(f"Clusters formed: {len(cluster_map)}")

    # =========================
    # 3️⃣ Assign IDs
    # =========================
    entities = []

    for i, (key, cluster) in enumerate(cluster_map.items(), start=1):

        # pick most common text as representative
        text_counts = {}
        for t in cluster["texts"]:
            text_counts[t] = text_counts.get(t, 0) + 1

        rep_text = max(text_counts, key=text_counts.get)

        entity = {
            "id": f"entity_{i:03d}",   # ⚠️ generic ID (not forcing shop)
            "text": rep_text,
            "norm_text": key,
            "instances": [
                {
                    "frame_id": inst["frame_id"],
                    "bbox": inst["bbox"]
                }
                for inst in cluster["instances"]
            ]
        }

        entities.append(entity)

    # =========================
    # 4️⃣ Attach to scene
    # =========================
    data["entities"] = entities

    # =========================
    # 5️⃣ Save
    # =========================
    output_path = os.path.join(output_root, "scene_with_entities.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved entity-merged scene: {output_path}")
    print(f"Total entities created: {len(entities)}\n")

    return output_path
