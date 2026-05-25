import os
from moviepy.editor import VideoFileClip


def cut_video_into_clips(input_video_path, output_directory,
                         clip_duration_seconds=60, fps=30):

    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Video not found: {input_video_path}")

    os.makedirs(output_directory, exist_ok=True)

    video = VideoFileClip(input_video_path)
    total_duration = int(video.duration)

    print(f"Total video duration: {total_duration} seconds")

    clip_count = 0
    start_time = 0
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]

    while start_time < total_duration:
        end_time = min(start_time + clip_duration_seconds, total_duration)

        clip = video.subclip(start_time, end_time)

        output_path = os.path.join(
            output_directory,
            f"{video_name}_clip_{clip_count:03d}.mp4"
        )

        print(f"Saving clip {clip_count}: {start_time}s → {end_time}s")

        clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=fps,
            verbose=False,
            logger=None
        )

        start_time = end_time
        clip_count += 1

    video.close()

    print(f"\nDone. {clip_count} clips saved in {output_directory}")

    return output_directory
