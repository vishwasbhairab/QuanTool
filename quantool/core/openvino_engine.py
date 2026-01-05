import os
import time
import nncf
from openvino.runtime import Core, serialize
from optimum.intel.openvino import OVModelForSequenceClassification


# ============================================================
# 1. EXPORT: Hugging Face → OpenVINO FP32
# ============================================================

def export_to_openvino(model_name: str, output_dir: str) -> None:
    """
    Export a Hugging Face sequence classification model to OpenVINO FP32 IR.

    Produces:
      - openvino_model.xml
      - openvino_model.bin
    """
    print("Exporting model to OpenVINO FP32 IR...")
    os.makedirs(output_dir, exist_ok=True)

    model = OVModelForSequenceClassification.from_pretrained(
        model_name,
        export=True,
        compile=False,
    )

    model.save_pretrained(output_dir)
    print(f"OpenVINO FP32 model saved to: {output_dir}")


# ============================================================
# 2. QUANTIZATION: OpenVINO FP32 → INT8 (NNCF)
# ============================================================

def quantize_openvino_int8(fp32_model_dir: str, output_dir: str) -> None:
    """
    Apply INT8 weight-only quantization using NNCF directly on OpenVINO model.

    Compatible with:
      - OpenVINO 2024.6
      - NNCF 2.8.0
    """
    print("Applying INT8 weight-only quantization via NNCF...")
    os.makedirs(output_dir, exist_ok=True)

    core = Core()

    # Load FP32 OpenVINO model
    model = core.read_model(
        model=os.path.join(fp32_model_dir, "openvino_model.xml"),
        weights=os.path.join(fp32_model_dir, "openvino_model.bin"),
    )

    # Apply INT8 weight compression
    compressed_model = nncf.compress_weights(
        model,
        mode=nncf.CompressWeightsMode.INT8_SYM,
    )

    # Save quantized model
    serialize(
        compressed_model,
        os.path.join(output_dir, "openvino_model.xml"),
        os.path.join(output_dir, "openvino_model.bin"),
    )

    print(f"OpenVINO INT8 model saved to: {output_dir}")

# ============================================================
# 2B. QUANTIZATION: OpenVINO FP32 → INT4 (Weight Compression)
# ============================================================

def compress_openvino_int4(fp32_model_dir: str, output_dir: str) -> None:
    """
    Apply INT4 weight-only compression using NNCF.

    NOTE:
    - Weights are compressed to INT4
    - Execution remains FP32
    - This reduces model size & memory footprint
    """
    print("Applying INT4 weight-only compression via NNCF...")
    os.makedirs(output_dir, exist_ok=True)

    core = Core()

    model = core.read_model(
        model=os.path.join(fp32_model_dir, "openvino_model.xml"),
        weights=os.path.join(fp32_model_dir, "openvino_model.bin"),
    )

    compressed_model = nncf.compress_weights(
        model,
        mode=nncf.CompressWeightsMode.INT4,
    )

    serialize(
        compressed_model,
        os.path.join(output_dir, "openvino_model.xml"),
        os.path.join(output_dir, "openvino_model.bin"),
    )

    print(f"OpenVINO INT4 (weight-compressed) model saved to: {output_dir}")


# ============================================================
# 3. INFERENCE ENGINE: OpenVINO Evaluator
# ============================================================

class OpenVINOEvaluator:
    """
    OpenVINO inference wrapper for latency benchmarking.
    """

    def __init__(self, model_dir: str, device: str = "CPU"):
        self.core = Core()

        self.model = self.core.read_model(
            model=os.path.join(model_dir, "openvino_model.xml")
        )

        self.compiled_model = self.core.compile_model(
            self.model, device
        )

        self.infer_request = self.compiled_model.create_infer_request()

    def infer(self, inputs: dict):
        """
        Run inference and measure latency.

        Args:
            inputs (dict): Tokenized inputs (input_ids, attention_mask)

        Returns:
            outputs: OpenVINO outputs
            latency_ms (float): Inference latency in milliseconds
        """
        start = time.time()
        outputs = self.infer_request.infer(inputs)
        latency_ms = (time.time() - start) * 1000
        return outputs, latency_ms
