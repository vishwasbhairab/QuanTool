import torch
import torch.backends.quantized

def check_qnnpack_support():
    """
    Checks if the 'qnnpack' quantized engine is supported in the current
    PyTorch installation and prints the result.
    """
    print("--- Checking PyTorch Quantization Backends ---")
    try:
        # Get the list of all supported quantized engines
        supported_engines = torch.backends.quantized.supported_engines
        print(f"Supported engines found: {supported_engines}")

        # Check if 'qnnpack' is in the list
        if 'qnnpack' in supported_engines:
            print("\n✅ Success: 'qnnpack' is available in your PyTorch installation.")
        else:
            print("\n❌ Error: 'qnnpack' is NOT available in your PyTorch installation.")
            print("Please reinstall PyTorch using the official command from pytorch.org.")

    except ImportError:
        print("\n❌ Error: PyTorch does not appear to be installed correctly.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    check_qnnpack_support()
