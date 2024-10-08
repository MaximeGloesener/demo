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
        #f"Quantization Type: {model_type}",
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
            annotated_frame = cv2.resize(annotated_frame, (1600, 900))
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
app.geometry("1920x1080")
app.title("YOLO benchmarking")
ctk.set_appearance_mode("light")

# Define button size
button_width = 250
button_height = 70

# Load images (ensure you have the correct paths to your images)
image1 = Image.open("yolov8.png").resize((450, 450))  # Resize to fit the layout
image2 = Image.open("pruning3.png").resize((480, 240))
image3 = Image.open("Quantization.png").resize((480, 240))
image4 = Image.open("KD.png").resize((480, 240))
logo = Image.open("Logos.png").resize((1500, 120))

# Convert images to PhotoImage format
photo1 = ImageTk.PhotoImage(image1)
photo2 = ImageTk.PhotoImage(image2)
photo3 = ImageTk.PhotoImage(image3)
photo4 = ImageTk.PhotoImage(image4)
logo_photo = ImageTk.PhotoImage(logo)

# Create 4 image labels for the images above the buttons
image_label1 = ctk.CTkLabel(app, image=photo1, text="")
image_label2 = ctk.CTkLabel(app, image=photo2, text="")
image_label3 = ctk.CTkLabel(app, image=photo3, text="")
image_label4 = ctk.CTkLabel(app, image=photo4, text="")
logo_label = ctk.CTkLabel(app, image=logo_photo, text="")

# Create 4 buttons; base model; fp16, int8, kd model
base_model_button = ctk.CTkButton(app, text="Base Model", font=("Arial Bold", 20, "bold" ), width=button_width, height=button_height, command=lambda: main("runs/detect/train3/weights/best.pt", "example_video.mp4", "results/jetson/yolo_m_results.json", "base_model"))
fp16_button = ctk.CTkButton(app, text="Light Compression", font=("Arial Bold", 20, "bold" ), width=button_width, height=button_height, command=lambda: main("runs/detect/train3/weights/best_fp16.engine", "example_video.mp4", "results/jetson/yolo_m_results.json", "fp16"))
int8_button = ctk.CTkButton(app, text="Medium Compression", font=("Arial Bold", 20, "bold" ),width=button_width, height=button_height, command=lambda: main("runs/detect/train3/weights/best_int8.engine", "example_video.mp4", "results/jetson/yolo_m_results.json", "int8"))
kd_button = ctk.CTkButton(app, text="High Compression",font=("Arial Bold", 20, "bold" ), width=button_width, height=button_height, command=lambda: main("runs/detect/train2/weights/best_fp16.pt", "example_video.mp4", "results/jetson/yolo_s_results.json", "fp16"))

# add title in row 0
title = ctk.CTkLabel(app, text="Object Detection in Railway Construction Sites", font=("Arial Bold", 50, "bold" )).grid(row=0, column=0, columnspan=4, padx=10, pady=10)

# add text in row 1
text1 = ctk.CTkLabel(app, text="Base Model", font=("Arial Bold", 30, "bold" )).grid(row=1, column=0, columnspan=1, padx=10, pady=50)
text2 = ctk.CTkLabel(app, text="Pruning", font=("Arial Bold", 30,  "bold")).grid(row=1, column=1, columnspan=1, padx=10, pady=50)
text3 = ctk.CTkLabel(app, text="Quantization", font=("Arial Bold", 30,  "bold")).grid(row=1, column=2, columnspan=1, padx=10, pady=50)
text4 = ctk.CTkLabel(app, text="Knowledge Distillation", font=("Arial Bold", 30,  "bold")).grid(row=1, column=3, columnspan=1, padx=10, pady=50)

# Place the images and buttons in a grid, side by side
image_label1.grid(row=2, column=0, padx=10, pady=10)
image_label2.grid(row=2, column=1, padx=10, pady=10)
image_label3.grid(row=2, column=2, padx=10, pady=10)
image_label4.grid(row=2, column=3, padx=10, pady=10)

base_model_button.grid(row=3, column=0, padx=10, pady=20)
fp16_button.grid(row=3, column=1, padx=10, pady=20)
int8_button.grid(row=3, column=2, padx=10, pady=20)
kd_button.grid(row=3, column=3, padx=10, pady=20)

logo_label.grid(row=4, column=0, columnspan=4, padx=10, pady=80)

# add exit button
exit_button = ctk.CTkButton(app, text="EXIT", font=("Arial Bold", 20, "bold" ), width=150, height=50, command=app.quit, fg_color="red", hover_color="#E64545")
exit_button.place(x=1750, y=20)

# Ensure the columns are of equal width to center the images and buttons
app.grid_columnconfigure((0, 1, 2, 3), weight=1)

# Run the main loop
app.mainloop()



