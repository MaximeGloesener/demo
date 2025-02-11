from ultralytics import YOLO
import argparse
import wandb

# Initialize Weights & Biases
wandb.init(project="yolov11-infrabel", config={
    "epochs": 100,
    "imgsz": 640,
    "batch": 16
})

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='yolov11n', help='model name')
args = parser.parse_args()

model = YOLO(f'{args.model}.pt')

# Enable W&B logging in YOLO training
model.train(data='data.yaml', epochs=100, imgsz=640, batch=16,
            project="yolov11-infrabel", name= f"{args.model}",
            save=True, val=True)

# Finish W&B run
wandb.finish()
