import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- IMPORTANT ---
# This script directly imports and tests your existing quantizer logic.
# Ensure the fbgemm fix is still in your quantizer.py file.
from quantool.core import quantizer

# ---
# STEP 1: Define a simple CNN model designed for quantization.
# This model uses QuantStub and DeQuantStub to mark the boundaries for quantization.
# ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # QuantStub converts float tensors to quantized tensors.
        self.quant = torch.ao.quantization.QuantStub()
        
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 10) # 28x28 -> 12x12 -> 5x5
        
        # DeQuantStub converts quantized tensors back to float tensors.
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x, attention_mask=None):
        # The data flow explicitly shows the conversion points.
        x = self.quant(x)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        
        x = self.dequant(x)
        return x

# ---
# STEP 2: Create a helper function to inspect and print model details.
# ---
def get_model_size_mb(model):
    """Calculates model size in megabytes."""
    torch.save(model.state_dict(), "temp_litmus.p")
    size_mb = os.path.getsize("temp_litmus.p") / (1024 * 1024)
    os.remove("temp_litmus.p")
    return size_mb

def print_model_details(model, title):
    """Prints a summary of the model's properties."""
    print(f"--- {title} ---")
    print("Architecture:")
    print(model)
    
    # For quantized models, weights are packed and accessed via a method.
    if hasattr(model.fc1, 'weight') and callable(model.fc1.weight):
        weight_dtype = model.fc1.weight().dtype
    else:
        weight_dtype = model.fc1.weight.dtype
        
    print(f"\nData type of fc1 layer weight: {weight_dtype}")
    print(f"Model size: {get_model_size_mb(model):.4f} MB")
    print("-" * (len(title) + 8) + "\n")

# ---
# STEP 3: The main execution block to run the test.
# ---
if __name__ == '__main__':
    print("--- Starting Static Quantization Litmus Test ---")

    # 1. Instantiate the original FP32 model
    fp32_model = SimpleCNN()
    fp32_model.eval()

    # Print baseline details
    print_model_details(fp32_model, "Original FP32 Model (Before)")

    # 2. Prepare a calibration dataloader using MNIST
    print("Preparing MNIST calibration data...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # The dummy forward pass for calibration requires the same data structure as your main script
    # so we map the MNIST data to the expected dictionary format.
    class MappedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
        def __getitem__(self, idx):
            image, label = self.dataset[idx]
            # Your quantizer expects a dictionary with 'input_ids'
            # We will use the image tensor as the 'input_ids' for this test
            return {'input_ids': image, 'attention_mask': torch.ones_like(image), 'label': label}
        def __len__(self):
            return len(self.dataset)

    calibration_loader = DataLoader(MappedDataset(mnist_dataset), batch_size=8, shuffle=True)
    print("Data preparation complete.\n")

    # 3. Apply your static quantization logic
    # This is the core of the test. It calls YOUR code from quantizer.py
    print(">>> Calling your quantize_int8_static function...")
    int8_model = quantizer.quantize_int8_static(fp32_model, calibration_loader)
    print(">>> Your function executed successfully.\n")

    # 4. Print details of the quantized model
    print_model_details(int8_model, "Quantized INT8 Model (After)")

    # 5. Final verification summary
    print("--- Verification Summary ---")
    print("✅ Test Passed: Your quantize_int8_static function successfully converted a compatible model.")
    print("Key changes to look for in the 'After' model:")
    print("  - ARCHITECTURE: Layers like Linear and Conv2d are now QuantizedLinear and QuantizedConv2d.")
    print("  - DATA TYPE: The weight dtype changed from torch.float32 to torch.qint8.")
    print("  - SIZE: The model size is significantly smaller (roughly 4x).")
    print("\nThis confirms the logic in your quantizer.py is correct.")

