import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import LOGGER

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=4):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_outputs, teacher_outputs):
        # Compute soft targets
        soft_targets = nn.functional.softmax(teacher_outputs / self.temperature, dim=-1)

        # Compute distillation loss
        distillation_loss = self.kl_div(
            nn.functional.log_softmax(student_outputs / self.temperature, dim=-1),
            soft_targets
        ) * (self.temperature ** 2)

        return distillation_loss

class DistillationTrainer(BaseTrainer):
    def __init__(self, teacher_model, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.distillation_loss = DistillationLoss()

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        images = batch['img']
        with torch.no_grad():
            teacher_outputs = self.teacher_model(images)
        batch['teacher_outputs'] = teacher_outputs
        return batch

    def criterion(self, preds, batch):
        loss = super().criterion(preds, batch)
        teacher_outputs = batch['teacher_outputs']
        distillation_loss = self.distillation_loss(preds[0], teacher_outputs[0])
        total_loss = loss + distillation_loss
        return total_loss

def train_with_distillation(teacher_model_path, student_model_path, data_yaml, epochs=50, imgsz=640, batch=32):
    # Load teacher model
    teacher_model = YOLO(teacher_model_path)
    teacher_model.model.eval()  # Set teacher model to evaluation mode

    # Load student model
    student_model = YOLO(student_model_path)

    # Create custom trainer
    trainer = DistillationTrainer(teacher_model=teacher_model.model, overrides={'model': student_model_path})

    # Train the student model with distillation
    results = student_model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        trainer=trainer
    )

    return student_model, results

# Usage
teacher_model_path = 'runs/detect/train/weights/best.pt'  # Path to the large model
student_model_path = 'runs/detect/train/weights/best.pt'  # Path to the small model
data_yaml = 'data.yaml'  # Path to your data configuration file

distilled_model, results = train_with_distillation(teacher_model_path, student_model_path, data_yaml)

# Save the distilled model
distilled_model.save('distilled_yolov8s.pt')