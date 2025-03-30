import torch

# Load the entire LightningModule
model = torch.load("model_full2.pth", map_location=torch.device('cpu' if not torch.cuda.is_available() else 'cuda'))
model.eval()  # Set to evaluation mode for inference



"""
from pytorch_bench import benchmark


results = benchmark(model, torch.randn(1, 3, 224, 224), gpu_only=True)


print(results)
"""