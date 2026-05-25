import os
import json
from bs4 import BeautifulSoup


def clean_text(text):
    """Normalize OCR text"""
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()


def is_valid_text(text):
    """Filter out noisy OCR detections"""
    if not text:
        return False

    # Too short
    if len(text) < 3:
        return False

    # Must contain at least 2 alphabet characters
    if sum(c.isalpha() for c in text) < 2:
        return False

    return True


def bbox_to_region(bbox, img_width=1280, img_height=720):
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    if x_center < img_width / 3:
        h_pos = "left"
    elif x_center > 2 * img_width / 3:
        h_pos = "right"
    else:
        h_pos = "center"

    if y_center < img_height / 3:
        v_pos = "upper"
    elif y_center > 2 * img_height / 3:
        v_pos = "lower"
    else:
        v_pos = "middle"

    return f"{v_pos}-{h_pos}"


def parse_ocr_folder(ocr_html_dir, output_root):

    parsed_dir = output_root
    os.makedirs(parsed_dir, exist_ok=True)

    html_files = sorted([
        f for f in os.listdir(ocr_html_dir)
        if f.endswith("_ocr_layout.html")
    ])

    for html_file in html_files:

        html_path = os.path.join(ocr_html_dir, html_file)

        # Extract frame ID (e.g., frame_00000)
        base_name = html_file.replace("_ocr_layout.html", "")
        frame_id = base_name

        with open(html_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        ocr_entries = []
        local_id_counter = 1  # per-frame temporary IDs

        for div in soup.find_all("div"):
            bbox = div.get("data-bbox")
            label = div.get("data-label")

            if bbox:
                try:
                    bbox = json.loads(bbox)
                except:
                    continue  # skip malformed bbox

                text = ""

                if div.find("p"):
                    text = div.find("p").get_text(separator=" ", strip=True)
                elif div.find("h1"):
                    text = div.find("h1").get_text(separator=" ", strip=True)
                elif div.find("img"):
                    text = div.find("img").get("alt", "")

                text = clean_text(text)

                if is_valid_text(text):
                    entry = {
                        "local_id": f"{frame_id}_det_{local_id_counter:03d}",
                        "frame_id": frame_id,
                        "text": text,
                        "norm_text": text.lower(),
                        "bbox": bbox,
                        "label": label,
                        "region": bbox_to_region(bbox)
                    }

                    ocr_entries.append(entry)
                    local_id_counter += 1

        json_name = f"{frame_id}_ocr.json"
        output_path = os.path.join(parsed_dir, json_name)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ocr_entries, f, indent=2, ensure_ascii=False)

        print(f"Saved: {output_path} ({len(ocr_entries)} detections)")

    return parsed_dir
