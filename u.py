# get model info layers params flops

import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='yolov8n.pt', help='model path')
args = parser.parse_args()

model = YOLO(f'{args.model}', task="detect")

model.fuse()
layers, params, gradients, flops = model.info()
print("benchmark results base model")
print('----------------------------')
print('Number of layers: ', layers)
print(f"Number of parameters: {params}")
print(f"Number of FLOPs: {flops}")


model_info = {
    'layers': layers,
    'params': params,
    'flops': flops
}
print(model_info)