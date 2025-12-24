"""
CLI runner for video detection + multi-object tracking.

Example:
python scripts/run_video.py \
  --video input.mp4 \
  --output_dir outputs \
  --tracker bytetrack \
  --batch_size 1 \
  --save_video \
  --save_csv \
  --save_json
"""

import os
import argparse
import csv
import json
import cv2

from detection.yolo_detector import YOLODetector
from tracking.bytetrack import ByteTracker
from inference.video_inference import VideoInferenceEngine
from visualization.draw_tracks import draw_tracks


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-object tracking on a video")

    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Detection batch size")

    parser.add_argument("--save_video", action="store_true", help="Save annotated video")
    parser.add_argument("--save_csv", action="store_true", help="Save CSV outputs")
    parser.add_argument("--save_json", action="store_true", help="Save JSON outputs")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize detector & tracker
    detector = YOLODetector()
    tracker = ByteTracker()

    engine = VideoInferenceEngine(
        detector=detector,
        tracker=tracker,
        batch_size=args.batch_size,
    )

    # Run inference
    results = engine.run(args.video)

    # Prepare video reader again for visualization
    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    video_writer = None
    if args.save_video:
        video_path = os.path.join(args.output_dir, "annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    csv_rows = []

    frame_idx = 0
    for frame_result in results:
        ret, frame = cap.read()
        if not ret:
            break

        objects = frame_result["objects"]

        if args.save_video:
            frame = draw_tracks(frame, objects)
            video_writer.write(frame)

        if args.save_csv:
            for obj in objects:
                csv_rows.append(
                    {
                        "frame_idx": frame_idx,
                        "id": obj["id"],
                        "x1": obj["bbox"][0],
                        "y1": obj["bbox"][1],
                        "x2": obj["bbox"][2],
                        "y2": obj["bbox"][3],
                        "confidence": obj["confidence"],
                    }
                )

        frame_idx += 1

    cap.release()
    if video_writer is not None:
        video_writer.release()

    # Write CSV
    if args.save_csv:
        csv_path = os.path.join(args.output_dir, "tracks.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame_idx",
                    "id",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "confidence",
                ],
            )
            writer.writeheader()
            writer.writerows(csv_rows)

    # Write JSON
    if args.save_json:
        json_path = os.path.join(args.output_dir, "tracks.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

    print("✅ Done. Outputs saved to:", args.output_dir)


if __name__ == "__main__":
    main()
