import json
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torchvision import transforms
from PIL import Image

from .model import build_model, load_trained_backbone


HERE = Path(__file__).resolve().parent
MODEL_WEIGHTS_PATH = HERE / "macro_expression_model.pth"
CLASSES_JSON_PATH = HERE / "macro_expression_classes.json"


_transform_infer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _load_classes() -> List[str]:
    if CLASSES_JSON_PATH.exists():
        with open(CLASSES_JSON_PATH, "r") as f:
            return json.load(f)
    # Fallback (dataset folders may differ slightly; prefer saved JSON)
    return ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]


def _prepare_image(image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    img_t = _transform_infer(image).unsqueeze(0)  # [1, 3, 224, 224]
    return img_t


def predict(image_path: str) -> Tuple[str, List[float]]:
    """
    Predict macro-expression for a single image.
    Returns (predicted_class_name, probabilities_list).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = _load_classes()
    model = build_model(len(classes))
    model.load_state_dict(torch.load(str(MODEL_WEIGHTS_PATH), map_location=device))
    model.to(device)
    model.eval()

    img_t = _prepare_image(image_path).to(device)
    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1)
        class_idx = int(torch.argmax(probs, dim=1).item())

    return classes[class_idx], probs.squeeze(0).cpu().tolist()


def embed(image_path: str, normalize: bool = False) -> List[float]:
    """
    Extract macro-expression embedding vector from the trained backbone.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, emb_dim = load_trained_backbone(str(MODEL_WEIGHTS_PATH), device=str(device))
    img_t = _prepare_image(image_path).to(device)
    with torch.no_grad():
        embedding = backbone(img_t).squeeze(0)  # [emb_dim]
        if normalize:
            embedding = torch.nn.functional.normalize(embedding, dim=0)
    return embedding.cpu().tolist()


def export_embedding_torchscript(output_path: Optional[str] = None):
    """
    Export the embedding backbone as a TorchScript module.
    The scripted module accepts a float tensor of shape [N, 3, 224, 224]
    and returns embeddings of shape [N, D].
    """
    if output_path is None:
        output_path = str(HERE / "macro_embedding.torchscript.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, _ = load_trained_backbone(str(MODEL_WEIGHTS_PATH), device=str(device))
    backbone.eval()
    backbone.to(device)

    example = torch.randn(1, 3, 224, 224, device=device)
    scripted = torch.jit.trace(backbone, example)
    torch.jit.save(scripted, output_path)
    return output_path


def export_embedding_onnx(output_path: Optional[str] = None):
    """
    Export the embedding backbone as an ONNX model (optional).
    """
    if output_path is None:
        output_path = str(HERE / "macro_embedding.onnx")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, _ = load_trained_backbone(str(MODEL_WEIGHTS_PATH), device=str(device))
    backbone.eval()
    backbone.to(device)

    example = torch.randn(1, 3, 224, 224, device=device)
    try:
        torch.onnx.export(
            backbone,
            example,
            output_path,
            input_names=["images"],
            output_names=["embeddings"],
            opset_version=12,
            dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
        )
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {e}")
    return output_path


# Example usage:
# pred_label, pred_probs = predict("path/to/image.jpg")
# embedding_vec = embed("path/to/image.jpg")
# ts_path = export_embedding_torchscript()
# onnx_path = export_embedding_onnx()


