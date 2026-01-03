import torch
from transformers import GPT2Tokenizer
from train_asterisk import AsteriskEmbeddingModel
import torch.onnx
import onnx
from onnxsim import simplify
from onnxruntime.quantization import quantize_dynamic, QuantType

def export_to_onnx(model_path="asterisk_embedding_model.pt", onnx_path="asterisk.onnx"):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'mask_token': '[MASK]'})
    model = AsteriskEmbeddingModel(vocab_size=len(tokenizer))
    model.token_emb = torch.nn.Embedding(len(tokenizer), 512)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randint(0, len(tokenizer), (1, 128))
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input_ids"],
        output_names=["sentence_embedding"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}},
        opset_version=18,
    )
    print(f"✅ Exported to {onnx_path}")

def simplify_onnx(onnx_path="asterisk.onnx", simplified_path="asterisk_simplified.onnx"):
    model = onnx.load(onnx_path)
    model_simp, check = simplify(model)
    if not check:
        raise RuntimeError("Simplified ONNX model could not be validated.")
    onnx.save(model_simp, simplified_path)
    print(f"✅ Simplified model saved to {simplified_path}")

def quantize_onnx(simplified_path="asterisk_simplified.onnx", quant_path="asterisk_int8.onnx"):
    quantize_dynamic(
        model_input=simplified_path,
        model_output=quant_path,
        weight_type=QuantType.QInt8
    )

    print(f"✅ Quantized model saved to {quant_path}")

if __name__ == "__main__":
    export_to_onnx()
    simplify_onnx()
    quantize_onnx()

