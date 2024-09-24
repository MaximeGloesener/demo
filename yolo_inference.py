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
    color = (255, 255, 255)  # yellow color
    thickness = 2
    line_spacing = 32

    model_info = stats['model_info']
    model_stats = stats[model_type]


    lines_top = [
        f"Model: {stats['model']}",
        f"Quantization Type: {model_type}",
        f"Layers: {model_info['layers']}, Params: {model_info['params']/1e6:.2f}M, FLOPS: {model_info['flops']:.2f}G",
        #f"FPS (GPU): {model_stats['fps_gpu']:.2f}",
        f'VRAM: {model_stats['max_memory_used']-model_stats['current_memory_used']:.2f}MB',
    ]

    lines_bottom = [
        f"Avg. Emissions: {model_stats['avg_emissions']*1e6:.2e} gCO2eq",
        f"Avg. Energy: {model_stats['avg_energy']*1e6:.2e} mWh",
        f"mAP50: {model_stats['map50']:.4f}, mAP50-95: {model_stats['map5095']:.4f}",
        f"Hardware: RTX 4090",
    ]
    for i, line in enumerate(lines_top):
        y = 30 + i * line_spacing
        cv2.putText(frame, line, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)

    for i, line in enumerate(lines_bottom):
        y = height - 30 - (len(lines_bottom) - i - 1) * line_spacing
        cv2.putText(frame, line, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return frame

def process_video(model, video_path, output_path, stats, model_type, conf_threshold=0.40, iou_threshold=0.7):
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
            results = model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Get FPS
            fps_gpu = results[0].speed["inference"]

            # add fps to the frame to right
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (255, 255, 255)
            thickness = 2
            y = 30
            cv2.putText(annotated_frame, f"FPS (GPU): {1000/fps_gpu:.2f}", (width-320, y), font, font_scale, color, thickness, cv2.LINE_AA)
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
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Benchmark YOLOv8')
    parser.add_argument('--model-path', type=str, default='best.pt', help='model name')
    parser.add_argument('--video-path', type=str, default='example_video.mp4', help='video name')
    parser.add_argument('--stats-path', type=str, default='yolo_n_results.json', help='stats name')
    args = parser.parse_args()

    model_path = args.model_path
    video_path = args.video_path
    stats_path = args.stats_path

    if 'fp16' in model_path: model_type = 'fp16'
    elif 'fp32' in model_path: model_type = 'fp32'
    elif 'int8' in model_path: model_type = 'int8'
    else: model_type = 'base_model'

    os.makedirs('output', exist_ok=True)
    output_path = f"output/out_video_{stats_path.split('.')[0].split('/')[2]}_{model_type}_{video_path.split('.')[0]}.mp4"
    main(model_path, video_path, output_path, stats_path, model_type)