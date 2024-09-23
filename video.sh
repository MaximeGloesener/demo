# generate video YOLOv8n inference: base model, fp32 engine, fp16, int8
python yolo_inference.py --model-path runs/detect/train/weights/best.pt --stats-path results/yolo_n_results.json
python yolo_inference.py --model-path runs/detect/train/weights/best_fp32.engine --stats-path results/yolo_n_results.json
python yolo_inference.py --model-path runs/detect/train/weights/best_fp16.engine --stats-path results/yolo_n_results.json
python yolo_inference.py --model-path runs/detect/train/weights/best_int8.engine --stats-path results/yolo_n_results.json


python yolo_inference.py --model-path runs/detect/train2/weights/best.pt --stats-path results/yolo_s_results.json
python yolo_inference.py --model-path runs/detect/train2/weights/best_fp32.engine --stats-path results/yolo_s_results.json
python yolo_inference.py --model-path runs/detect/train2/weights/best_fp16.engine --stats-path results/yolo_s_results.json
python yolo_inference.py --model-path runs/detect/train2/weights/best_int8.engine --stats-path results/yolo_s_results.json


python yolo_inference.py --model-path runs/detect/train3/weights/best.pt --stats-path results/yolo_m_results.json
python yolo_inference.py --model-path runs/detect/train3/weights/best_fp32.engine --stats-path results/yolo_m_results.json
python yolo_inference.py --model-path runs/detect/train3/weights/best_fp16.engine --stats-path results/yolo_m_results.json
python yolo_inference.py --model-path runs/detect/train3/weights/best_int8.engine --stats-path results/yolo_m_results.json


python yolo_inference.py --model-path runs/detect/train4/weights/best.pt --stats-path results/yolo_l_results.json
python yolo_inference.py --model-path runs/detect/train4/weights/best_fp32.engine --stats-path results/yolo_l_results.json
python yolo_inference.py --model-path runs/detect/train4/weights/best_fp16.engine --stats-path results/yolo_l_results.json
python yolo_inference.py --model-path runs/detect/train4/weights/best_int8.engine --stats-path results/yolo_l_results.json


python yolo_inference.py --model-path runs/detect/train5/weights/best.pt --stats-path results/yolo_x_results.json
python yolo_inference.py --model-path runs/detect/train5/weights/best_fp32.engine --stats-path results/yolo_x_results.json
python yolo_inference.py --model-path runs/detect/train5/weights/best_fp16.engine --stats-path results/yolo_x_results.json
python yolo_inference.py --model-path runs/detect/train5/weights/best_int8.engine --stats-path results/yolo_x_results.json