from ultralytics import YOLO
from benchmark_yolo import benchmark
import argparse
import json
import os

parser = argparse.ArgumentParser(description='Benchmark YOLOv8')
parser.add_argument('--model-path', type=str, default='runs/detect/train/weights/best.pt', help='model name')
args = parser.parse_args()

engine_path = '/'.join(args.model_path.split('/')[:-1])+'/best.engine'
results_path = args.model_path.split('/')[-3]

# base model full precision non pruned
model_weight = args.model_path


all_results = {}
all_results['model'] = model_weight

# load the model
base_model = YOLO(model_weight, task="detect")
print('benchmarking base model')
results = benchmark(base_model, 640)
all_results['base_model'] = results
# save all the results to a json file for later comparison


# export yolo models to tensorrt and apply quantization
# TensorRT FP32
print('starting fp 32')
base_model.export(format="engine",
                   imgsz=640,
                   dynamic=True,
                   verbose=False,
                   workspace=2,
                   batch=8)
model = YOLO(engine_path, task="detect")
results = benchmark(model, 640, type='engine')
all_results['fp32'] = results
os.rename(engine_path, engine_path.replace('.engine', '_fp32.engine'))


# TensorRT FP16
print('starting fp 16')
base_model.export(format="engine",
                   imgsz=640,
                   dynamic=True,
                   verbose=False,
                   batch=8,
                   workspace=2,
                   half=True)
model = YOLO(engine_path, task="detect")
results = benchmark(model, 640, type='engine')
all_results['fp16'] = results
os.rename(engine_path, engine_path.replace('.engine', '_fp16.engine'))

# TensorRT INT8 with calibration `data`
print('starting int 8')
base_model.export(format="engine",
                   imgsz=640,
                   dynamic=True,
                   verbose=False,
                   batch=8,
                   int8=True,
                   workspace=2,
                   data="data.yaml")
model = YOLO(engine_path, task="detect")
results = benchmark(model, 640, type='engine')
all_results['int8'] = results
os.rename(engine_path, engine_path.replace('.engine', '_int8.engine'))

# save all the results to a json file for later comparison
with open(f'results{results_path}.json', 'w') as f:
    json.dump(all_results, f)