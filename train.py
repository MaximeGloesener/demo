from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='yolov8n', help='model name')
args = parser.parse_args()


model = YOLO(f'{args.model}.pt')
model.train(data='data.yaml', epochs=50, imgsz=640, batch=16)
