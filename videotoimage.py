
import os
import cv2
import argparse

QVGA_WIDTH, QVGA_HEIGHT = 320, 240

def extract_frames(video_path: str, output_dir: str, img_format: str, start_index: int):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frame_idx = start_index
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_qvga = cv2.resize(frame, (QVGA_WIDTH, QVGA_HEIGHT))

        filename = os.path.join(
            output_dir,
            f"frame_{frame_idx:06d}.{img_format}"
        )
        cv2.imwrite(filename, frame_qvga)
        frame_idx += 1

    cap.release()
    print(f"Extracted and resized {frame_idx - start_index} frames to '{output_dir}' at {QVGA_WIDTH}×{QVGA_HEIGHT}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract all frames from a video into QVGA (320×240) images."
    )
    parser.add_argument(
        "video_path",
        help="Path to the input video file (e.g. input.mp4)"
    )
    parser.add_argument(
        "output_dir",
        help="Directory where extracted frames will be saved"
    )
    parser.add_argument(
        "--format",
        default="jpg",
        choices=["jpg", "png", "bmp"],
        help="Image format for output frames (default: jpg)"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index for frame numbering (default: 0)"
    )
    args = parser.parse_args()

    extract_frames(
        video_path=args.video_path,
        output_dir=args.output_dir,
        img_format=args.format,
        start_index=args.start_index
    )

if __name__ == "__main__":
    main()
