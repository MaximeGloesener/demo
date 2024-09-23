from ultralytics import YOLO
import cv2
from tqdm import tqdm

def load_model(model_path):
    return YOLO(model_path)

def process_video(model, video_path, output_path, conf_threshold=0.3, iou_threshold=0.6):
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

            # Write the frame
            out.write(annotated_frame)

            pbar.update(1)

    # Release the video capture object and writer
    video.release()
    out.release()

def main(model_path, video_path, output_path):
    # Load the model
    model = load_model(model_path)

    # Process the video
    process_video(model, video_path, output_path)

if __name__ == "__main__":
    model_path = "best.pt"
    video_path = "example_video.mp4"
    output_path = "out_video.mp4"
    main(model_path, video_path, output_path)