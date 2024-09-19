# export yolo models to tensorrt and apply quantization
# see doc: https://docs.ultralytics.com/integrations/tensorrt/

from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='yolov8n', help='model name')
args = parser.parse_args()



model = YOLO(args.model)

# TensorRT FP32
out = model.export(format="engine",
                   imgsz=640,
                   dynamic=True,
                   verbose=False,
                   batch=8)

# TensorRT FP16
out = model.export(format="engine",
                   imgsz=640,
                   dynamic=True,
                   verbose=False,
                   batch=8,
                   half=True)

# TensorRT INT8 with calibration `data`
out = model.export(format="engine",
                   imgsz=640,
                   dynamic=True,
                   verbose=False,
                   batch=8,
                   int8=True,
                   data="data.yaml")