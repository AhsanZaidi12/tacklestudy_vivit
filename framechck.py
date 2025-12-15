import os
import cv2

def count_frames_in_videos(folder_path):
    video_found = False
    non_32_videos = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_found = True
                video_path = os.path.join(root, file)
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"❌ Failed to open: {video_path}")
                    continue
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count != 32:
                    non_32_videos.append((video_path, frame_count))
                cap.release()

    if not video_found:
        print("⚠️ No video files found in the given directory.")
    elif not non_32_videos:
        print("✅ All videos have exactly 32 frames.")
    else:
        print("📌 Videos with frame count not equal to 32:")
        for path, count in non_32_videos:
            print(f"{path}: {count} frames")

# Replace with your actual folder path
count_frames_in_videos("/mnt/e/Tackle_Ablation/videos")
