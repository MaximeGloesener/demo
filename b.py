from ultralytics import YOLO
import argparse
import json

parser = argparse.ArgumentParser(description='Benchmark YOLOv8')
parser.add_argument('--model-path', type=str, default='runs/detect/train/weights/best.pt', help='model name')
args = parser.parse_args()

model = YOLO(args.model_path, task="detect")
model.fuse()
layers, params, gradients, flops = model.info()
print("benchmark results base model")
print('----------------------------')
print('Number of layers: ', layers)
print(f"Number of parameters: {params}")
print(f"Number of FLOPs: {flops}")

all_results = {}
all_results['model'] = args.model_path
all_results['base_model'] = {
    'layers': layers,
    'params': params,
    'flops': flops
}

results_path = args.model_path.split('/')[-3]
# save all the results to a json file for later comparison
with open(f'results_{results_path}.json', 'w') as f:
    json.dump(all_results, f)