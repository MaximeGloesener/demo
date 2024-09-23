from ultralytics import YOLO
import cv2
from tqdm import tqdm
import json

def load_model(model_path):
    return YOLO(model_path)

def load_stats(stats_path):
    with open(stats_path, 'r') as f:
        return json.load(f)

def add_stats_to_frame(frame, stats, model_type):
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  # White color
    thickness = 2
    line_spacing = 30

    model_info = stats['model_info']
    model_stats = stats[model_type]

    lines = [
        f"Model: {stats['model']}",
        f"Type: {model_type}",
        f"Layers: {model_info['layers']}, Params: {model_info['params']/1e6}M, FLOPS: {model_info['flops']:.2f}G",
        f"FPS (GPU): {model_stats['fps_gpu']:.2f}",
        f"mAP50: {model_stats['map50']:.4f}, mAP50-95: {model_stats['map5095']:.4f}",
        f"Avg. Emissions: {model_stats['avg_emissions']:.2e} kgCO2eq",
        f"Avg. Energy: {model_stats['avg_energy']:.2e} kWh"
    ]

    for i, line in enumerate(lines):
        y = 30 + i * line_spacing
        cv2.putText(frame, line, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return frame

def process_video(model, video_path, output_path, stats, model_type, conf_threshold=0.3, iou_threshold=0.6):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process the video
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            # Run YOLOv8 inference on the frame
            results = model(frame, conf=conf_threshold, iou=iou_threshold)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Add stats to the frame
            annotated_frame = add_stats_to_frame(annotated_frame, stats, model_type)

            # Write the frame
            out.write(annotated_frame)

            pbar.update(1)

    # Release the video capture object and writer
    video.release()
    out.release()

def main(model_path, video_path, output_path, stats_path, model_type):
    # Load the model
    model = load_model(model_path)

    # Load the stats
    stats = load_stats(stats_path)

    # Process the video
    process_video(model, video_path, output_path, stats, model_type)

if __name__ == "__main__":
    model_path = "best.pt"
    video_path = "example_video.mp4"
    stats_path = "resultstrain.json"

    # Process for each model type
    for model_type in ["base_model"]:
        output_path = f"out_video_{model_type}.mp4"
        main(model_path, video_path, output_path, stats_path, model_type)