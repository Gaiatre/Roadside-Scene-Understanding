import cv2
import os


def extract_frames(video_path, output_root, frames_per_second=1):

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(output_root, video_name, "frames")

    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if video_fps == 0:
        raise RuntimeError("Video FPS could not be determined.")

    frame_interval = max(1, int(video_fps / frames_per_second))

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_path = os.path.join(
                frames_dir,
                f"frame_{saved_count:05d}.jpg"
            )
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()

    print(f"Done. Saved {saved_count} frames to {frames_dir}")
    return frames_dir
