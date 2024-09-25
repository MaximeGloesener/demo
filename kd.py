import torch
from ultralytics import YOLO

# Load the trained teacher model (YOLOv8x)
teacher_model = YOLO('runs/detect/train5/weights/best.pt')

# Load the student model (YOLOv8s)
student_model = YOLO('runs/detect/train/weights/best.pt')

# Define the loss function for distillation
def distillation_loss(student_outputs, teacher_outputs, target, alpha=0.5, temperature=3):
    # Standard loss (e.g., YOLO loss)
    student_loss = student_model.compute_loss(student_outputs, target)

    # Distillation loss: compare teacher and student outputs
    soft_teacher = torch.nn.functional.softmax(teacher_outputs / temperature, dim=1)
    soft_student = torch.nn.functional.log_softmax(student_outputs / temperature, dim=1)

    distill_loss = torch.nn.functional.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)

    # Combined loss
    return alpha * student_loss + (1 - alpha) * distill_loss

# Training loop for student model using distillation
def train_student_with_kd(student_model, teacher_model, data, epochs, imgsz, batch_size):
    # Dataloader and optimizer
    dataloader = student_model.dataloader(data, imgsz=imgsz, batch_size=batch_size)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        student_model.train()

        for batch in dataloader:
            imgs, targets = batch

            # Forward pass of student
            student_outputs = student_model(imgs)

            # Get teacher's outputs (logits)
            with torch.no_grad():
                teacher_outputs = teacher_model(imgs)

            # Compute knowledge distillation loss
            loss = distillation_loss(student_outputs, teacher_outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Train the student model with knowledge distillation
train_student_with_kd(student_model, teacher_model, data='data.yaml', epochs=50, imgsz=640, batch_size=32)
