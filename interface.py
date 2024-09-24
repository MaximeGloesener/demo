import customtkinter as ctk
from PIL import Image, ImageTk
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
    color = (255, 255, 255)  # white color
    thickness = 2
    line_spacing = 32

    model_info = stats['model_info']
    model_stats = stats[model_type]

    lines_top = [
        f"Model: {stats['model']}",
        f"Quantization Type: {model_type}",
        f"Layers: {model_info['layers']}, Params: {model_info['params']/1e6:.2f}M, FLOPS: {model_info['flops']:.2f}G",
        # f'VRAM: {model_stats["max_memory_used"]-model_stats["current_memory_used"]:.2f}MB',
    ]

    lines_bottom = [
        f"Avg. Emissions: {model_stats['avg_emissions']*1e6:.2e} gCO2eq",
        f"Avg. Energy: {model_stats['avg_energy']*1e6:.2e} mWh",
        f"mAP50: {model_stats['map50']:.4f}, mAP50-95: {model_stats['map5095']:.4f}",
        f"Hardware: Jetson Xavier",
    ]

    for i, line in enumerate(lines_top):
        y = 30 + i * line_spacing
        cv2.putText(frame, line, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)

    for i, line in enumerate(lines_bottom):
        y = height - 30 - (len(lines_bottom) - i - 1) * line_spacing
        cv2.putText(frame, line, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return frame

def process_video(model, video_path, stats, model_type, conf_threshold=0.40, iou_threshold=0.7):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process the video and display each frame
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

            # Add FPS to the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (255, 255, 255)
            thickness = 2
            y = 30
            cv2.putText(annotated_frame, f"FPS (GPU): {1000/fps_gpu:.2f}", (width-320, y), font, font_scale, color, thickness, cv2.LINE_AA)

            # Add stats to the frame
            annotated_frame = add_stats_to_frame(annotated_frame, stats, model_type)

            # Display the frame
            # resize the frame to fit the screen
            annotated_frame = cv2.resize(annotated_frame, (1280, 720))
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            pbar.update(1)

    # Release the video capture object
    video.release()
    cv2.destroyAllWindows()

def main(model_path, video_path, stats_path, model_type):
    # Load the model
    model = load_model(model_path)

    # Load the stats
    stats = load_stats(stats_path)

    # Process the video and display it
    process_video(model, video_path, stats, model_type)



# Initialize the main window
app = ctk.CTk()
app.geometry("1080x400")
app.title("YOLO benchmarking")
ctk.set_appearance_mode("dark")

# Define button size
button_width = 200
button_height = 50

# Load images (ensure you have the correct paths to your images)
image1 = Image.open("image.jpg").resize((250, 250))  # Resize to fit the layout
image2 = Image.open("image.jpg").resize((250, 250))
image3 = Image.open("image.jpg").resize((250, 250))
image4 = Image.open("image.jpg").resize((250, 250))

# Convert images to PhotoImage format
photo1 = ImageTk.PhotoImage(image1)
photo2 = ImageTk.PhotoImage(image2)
photo3 = ImageTk.PhotoImage(image3)
photo4 = ImageTk.PhotoImage(image4)

# Create 4 image labels for the images above the buttons
image_label1 = ctk.CTkLabel(app, image=photo1, text="")
image_label2 = ctk.CTkLabel(app, image=photo2, text="")
image_label3 = ctk.CTkLabel(app, image=photo3, text="")
image_label4 = ctk.CTkLabel(app, image=photo4, text="")


# when you click on button, run YOLO inference on the selected model and display stats on the image
def run_inference():
    print(f"Running inference")
    main("best.pt", "example_video.mp4", "results/4090/yolo_n_results.json", "base_model")


# Create 4 buttons; base model; fp16, int8, kd model
base_model_button = ctk.CTkButton(app, text="Base Model", width=button_width, height=button_height, command=lambda: main("runs/detect/train5/weights/best.pt", "example_video.mp4", "results/jetson/yolo_x_results.json", "base_model"))
fp16_button = ctk.CTkButton(app, text="FP16 Model", width=button_width, height=button_height, command=lambda: main("runs/detect/train5/weights/best_fp16.engine", "example_video.mp4", "results/jetson/yolo_x_results.json", "fp16"))
int8_button = ctk.CTkButton(app, text="INT8 Model", width=button_width, height=button_height, command=lambda: main("runs/detect/train5/weights/best_int8.engine", "example_video.mp4", "results/jetson/yolo_x_results.json", "int8"))
kd_button = ctk.CTkButton(app, text="Knowledge Distilled Model", width=button_width, height=button_height, command=lambda: main("runs/detect/train5/weights/best_kd.pt", "example_video.mp4", "results/jetson/yolo_x_results.json", "kd"))

# Place the images and buttons in a grid, side by side
image_label1.grid(row=0, column=0, padx=10, pady=10)
image_label2.grid(row=0, column=1, padx=10, pady=10)
image_label3.grid(row=0, column=2, padx=10, pady=10)
image_label4.grid(row=0, column=3, padx=10, pady=10)

base_model_button.grid(row=1, column=0, padx=10, pady=20)
fp16_button.grid(row=1, column=1, padx=10, pady=20)
int8_button.grid(row=1, column=2, padx=10, pady=20)
kd_button.grid(row=1, column=3, padx=10, pady=20)

# Ensure the columns are of equal width to center the images and buttons
app.grid_columnconfigure((0, 1, 2, 3), weight=1)

# Run the main loop
app.mainloop()



