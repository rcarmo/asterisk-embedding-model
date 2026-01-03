from transformers import AutoTokenizer, AutoModel
from optimum.onnxruntime import ORTModelForFeatureExtraction
from optimum.exporters.onnx import main_export

model_id = "sentence-transformers/all-MiniLM-L6-v2"

# Export to ONNX
main_export(
    model_name_or_path=model_id,
    output="onnx_model",
    task="feature-extraction",
    opset=12
)

