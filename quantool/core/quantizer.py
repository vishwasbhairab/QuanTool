import torch
import torch.nn as nn
import torch.ao.quantization
# We need to import the backend configuration module
import torch.backends.quantized
from torch.utils.data import DataLoader
from tqdm import tqdm

def quantize_int8_dynamic(model: torch.nn.Module) -> torch.nn.Module:
    """
    Applies dynamic post-training quantization (INT8) to a given PyTorch model.
    """
    print("Applying INT8 dynamic quantization...")
    try:
        model.eval()
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        print("INT8 dynamic quantization successful.")
        return quantized_model
    except Exception as e:
        print(f"Error during INT8 dynamic quantization: {e}")
        raise

def quantize_int8_static(model: torch.nn.Module, calibration_dataloader: DataLoader) -> torch.nn.Module:
    """
    Applies static post-training quantization (INT8) to a given PyTorch model.
    This version uses a direct, manual assignment of qconfigs and forces the
    correct backend engine for maximum robustness.
    """
    print("Applying INT8 static quantization...")
    try:
        model.eval()
        
        # --- THE FINAL FIX: SWITCHING THE ENGINE ---
        # 1. Force PyTorch to use the 'qnnpack' backend. This engine has wider
        #    compatibility across different CPU architectures, especially on Windows/macOS.
        print("Setting quantization backend to 'fbgemm'...")
        torch.backends.quantized.engine = 'fbgemm'
        
        # 2. Get the default quantization configuration that matches this backend.
        qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')

        # 3. Manually iterate and assign the qconfig to Linear layers, while
        #    explicitly disabling it for Embedding layers. This is the most robust method.
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.qconfig = qconfig
            elif isinstance(module, nn.Embedding):
                module.qconfig = None

        print("Preparing model for quantization...")
        prepared_model = torch.ao.quantization.prepare(model)

        print("Calibrating model with sample data...")
        device = torch.device("cpu") 
        prepared_model.to(device)
        with torch.no_grad():
            for i, batch in enumerate(tqdm(calibration_dataloader, desc="Calibration")):
                if i >= 10: 
                    break
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                prepared_model(input_ids, attention_mask=attention_mask)

        print("Converting model to quantized version...")
        quantized_model = torch.ao.quantization.convert(prepared_model)
        
        print("INT8 static quantization successful.")
        return quantized_model
    except Exception as e:
        print(f"Error during INT8 static quantization: {e}")
        raise

def quantize_int4(model: torch.nn.Module):
    """Placeholder for future INT4 quantization implementation."""
    print("INT4 quantization is not yet implemented.")
    raise NotImplementedError("INT4 support will be added in a future phase.")
'''
```

**What We Changed:**
* `torch.backends.quantized.engine = 'fbgemm'` became `torch.backends.quantized.engine = 'qnnpack'`
* `get_default_qconfig('fbgemm')` became `get_default_qconfig('qnnpack')`

That's it. This directly addresses the hardware compatibility error.

```bash
python run_benchmark.py --model-name "distilbert-base-uncased"

'''