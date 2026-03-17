"""
ECG Classifier Serving Container.

Loads MedSigLIP (google/medsiglip-448) + MoE/MLP classifier checkpoint
and serves predictions via HTTP.

Routes:
  - POST /predict      — classify a base64-encoded ECG image
  - POST /embed/image  — return the MedSigLIP image embedding
  - POST /score        — run image -> MedSigLIP -> classifier scoring
  - GET  /health    — liveness / readiness check

The model checkpoint is resolved in this order:
  1. AIP_STORAGE_URI env var (set automatically by Vertex AI)
  2. CLASSIFIER_CKPT_PATH env var
  3. CHECKPOINT_PATH env var (backward compatibility)
  4. Default path relative to this script

Port is resolved as: PORT → AIP_HTTP_PORT → 8080.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model architectures (must match training code)
# ---------------------------------------------------------------------------

class ExpertMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: tuple[int, ...] = (1028, 512, 256),
        dropout: tuple[float, ...] = (0.15, 0.15, 0.10),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        dropout_values = tuple(float(p) for p in dropout)
        if len(dropout_values) < len(hidden):
            dropout_values = dropout_values + (0.0,) * (len(hidden) - len(dropout_values))
        elif len(dropout_values) > len(hidden):
            dropout_values = dropout_values[: len(hidden)]
        for h, p in zip(hidden, dropout_values):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(p))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoEClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_experts: int = 5,
        gate_hidden: int = 512,
        temperature: float = 1.0,
        expert_hidden: tuple[int, ...] = (1028, 512, 256),
        expert_dropout: tuple[float, ...] = (0.15, 0.15, 0.10),
    ):
        super().__init__()
        self.temperature = temperature
        self.experts = nn.ModuleList(
            [
                ExpertMLP(in_dim, out_dim, hidden=expert_hidden, dropout=expert_dropout)
                for _ in range(num_experts)
            ]
        )
        self.gate = nn.Sequential(
            nn.Linear(in_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, num_experts),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_logits = self.gate(x) / self.temperature
        gate_w = torch.softmax(gate_logits, dim=-1)
        expert_logits = torch.stack([expert(x) for expert in self.experts], dim=1)
        mixed_logits = torch.sum(expert_logits * gate_w.unsqueeze(-1), dim=1)
        return mixed_logits, gate_w, expert_logits


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_1: int, hidden_2: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.out = nn.Linear(hidden_2, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def build_classifier(ckpt: dict[str, Any]) -> tuple[nn.Module, str]:
    """Reconstruct classifier architecture from checkpoint metadata."""
    state_dict = ckpt.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError("Checkpoint missing state_dict.")

    embed_dim = int(ckpt["embed_dim"])
    num_classes = int(ckpt["num_classes"])

    if any(key.startswith("experts.") for key in state_dict):
        num_experts = int(ckpt.get("num_experts", 5))
        expert_linear_layers: list[tuple[int, torch.Tensor]] = []
        for key, value in state_dict.items():
            if (
                key.startswith("experts.0.net.")
                and key.endswith(".weight")
                and isinstance(value, torch.Tensor)
                and value.ndim == 2
            ):
                layer_index = int(key.split(".")[3])
                expert_linear_layers.append((layer_index, value))

        if len(expert_linear_layers) < 2:
            raise RuntimeError("Unable to infer expert architecture from checkpoint.")

        expert_linear_layers.sort(key=lambda item: item[0])
        expert_hidden = tuple(int(w.shape[0]) for _, w in expert_linear_layers[:-1])
        gate_hidden = (
            int(state_dict["gate.0.weight"].shape[0])
            if "gate.0.weight" in state_dict
            else 256
        )
        model = MoEClassifier(
            in_dim=embed_dim,
            out_dim=num_classes,
            num_experts=num_experts,
            gate_hidden=gate_hidden,
            temperature=1.0,
            expert_hidden=expert_hidden,
            expert_dropout=tuple(0.0 for _ in expert_hidden),
        )
        model_type = "moe"
    elif {"fc1.weight", "fc2.weight", "out.weight"}.issubset(state_dict):
        hidden_1 = int(state_dict["fc1.weight"].shape[0])
        hidden_2 = int(state_dict["fc2.weight"].shape[0])
        model = MLPClassifier(embed_dim, hidden_1, hidden_2, num_classes)
        model_type = "mlp"
    else:
        raise RuntimeError("Unsupported checkpoint format.")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, model_type


def extract_features(output: Any) -> torch.Tensor:
    """Extract image features from MedSigLIP output."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0, :]
    raise TypeError(f"Unexpected feature output type: {type(output)}")


# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------------------

_embedder = None
_processor = None
_classifier = None
_classifier_type = ""
_device = "cpu"
_classes: list[str] = []
_threshold: float = 0.5
_model_id = ""
_embed_dim = 0
_num_experts = 0


def _resolve_checkpoint_path() -> Path:
    """
    Find the checkpoint file.

    Priority:
    1. AIP_STORAGE_URI (set by Vertex AI) — the GCS model artifact directory
    2. CLASSIFIER_CKPT_PATH env var
    3. CHECKPOINT_PATH env var (local dev fallback)
    4. Default path relative to this script
    """
    # Vertex AI mounts GCS artifacts here
    storage_uri = os.environ.get("AIP_STORAGE_URI", "").strip()
    if storage_uri:
        # Vertex AI downloads GCS artifacts to a local directory
        # Look for .pt files in the directory
        storage_path = Path(storage_uri)
        if storage_path.is_dir():
            pt_files = list(storage_path.glob("*.pt"))
            if pt_files:
                return pt_files[0]

    # Local fallback
    local_path = os.environ.get("CLASSIFIER_CKPT_PATH", "").strip()
    if local_path:
        return Path(local_path)

    local_path = os.environ.get("CHECKPOINT_PATH", "").strip()
    if local_path:
        return Path(local_path)

    # Default: look next to this script
    default = Path(__file__).parent.parent / "embed_data" / "moe_classifier_medsiglip.pt"
    return default


def load_models():
    """Load MedSigLIP embedder + classifier at startup."""
    global _embedder, _processor, _classifier, _classifier_type
    global _device, _classes, _threshold, _model_id, _embed_dim, _num_experts

    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = os.environ.get("MEDSIGLIP_MODEL_ID", "google/medsiglip-448").strip()
    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    checkpoint_path = _resolve_checkpoint_path()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(
        "Loading models: model_id=%s checkpoint=%s device=%s",
        model_id,
        checkpoint_path,
        device,
    )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Invalid checkpoint structure.")

    # Load MedSigLIP
    embedder = AutoModel.from_pretrained(model_id, token=hf_token)
    processor = AutoImageProcessor.from_pretrained(model_id, token=hf_token)
    embedder.to(device)
    embedder.eval()

    # Load classifier
    classifier, classifier_type = build_classifier(ckpt)
    classifier.to(device)
    classifier.eval()

    # Extract metadata
    num_classes = int(ckpt["num_classes"])
    classes = ckpt.get("classes")
    if not isinstance(classes, list) or len(classes) != num_classes:
        classes = [f"class_{i}" for i in range(num_classes)]

    _embedder = embedder
    _processor = processor
    _classifier = classifier
    _classifier_type = {
        "moe": "moe_classifier",
        "mlp": "mlp_classifier",
    }.get(classifier_type, classifier_type)
    _device = device
    _classes = [str(c) for c in classes]
    _threshold = float(ckpt.get("threshold", 0.5))
    _model_id = model_id
    _embed_dim = int(ckpt["embed_dim"])
    _num_experts = int(ckpt.get("num_experts") or 0)

    logger.info(
        "Models loaded: classifier_type=%s classes=%s embed_dim=%s threshold=%.3f device=%s",
        _classifier_type,
        _classes,
        _embed_dim,
        _threshold,
        _device,
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="ECG Classifier", version="1.0.0")


@app.on_event("startup")
def startup():
    load_models()


class PredictRequest(BaseModel):
    image_base64: str


class PredictResponse(BaseModel):
    classifier_type: str
    medsiglip_model_id: str
    classes: list[str]
    scores: list[float]
    scores_by_class: dict[str, float]
    predicted_labels: list[str]
    threshold: float


class EmbedImageResponse(BaseModel):
    embedding: list[float]


class ScoreResult(BaseModel):
    index: int
    embedding: list[float]
    scores: list[float]
    scores_by_class: dict[str, float]
    probabilities: list[float]
    probabilities_by_class: dict[str, float]
    predictions: list[int]
    predictions_by_class: dict[str, int]
    predicted_labels: list[str]
    gate_weights: list[float]


class ScoreResponse(BaseModel):
    classifier_type: str
    medsiglip_model_id: str
    device: str
    classes: list[str]
    threshold: float
    num_experts: int
    results: list[ScoreResult]
    scoring_mode: str


@app.get("/health")
def health():
    """Health check for Vertex AI."""
    return {
        "status": "healthy" if _embedder is not None else "loading",
        "classifier_type": _classifier_type,
        "device": _device,
    }


def _ensure_models_loaded() -> None:
    if _embedder is None or _processor is None or _classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")


def _load_image_from_bytes(raw: bytes) -> Image.Image:
    if not raw:
        raise HTTPException(status_code=400, detail="Image file is empty.")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


def _decode_base64_image(payload: str) -> Image.Image:
    encoded = (payload or "").strip()
    if not encoded:
        raise HTTPException(status_code=400, detail="image_base64 is empty.")
    if encoded.startswith("data:") and "," in encoded:
        encoded = encoded.split(",", 1)[1]
    try:
        raw = base64.b64decode(encoded)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}") from exc
    return _load_image_from_bytes(raw)


def _embed_image(image: Image.Image) -> torch.Tensor:
    inputs = _processor(images=[image], return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    image_features = extract_features(_embedder.get_image_features(**inputs))

    if image_features.ndim != 2 or image_features.shape[0] != 1:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected embedding shape: {tuple(image_features.shape)}",
        )
    if int(image_features.shape[1]) != _embed_dim:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding dim mismatch: expected={_embed_dim}, got={int(image_features.shape[1])}",
        )
    return image_features


def _score_embedding(image_features: torch.Tensor) -> ScoreResult:
    logits_output = _classifier(image_features)
    gate_weights: list[float] = []
    if isinstance(logits_output, tuple):
        logits = logits_output[0]
        if len(logits_output) > 1:
            gate_weights = logits_output[1].squeeze(0).detach().cpu().tolist()
    else:
        logits = logits_output

    raw_scores = logits.squeeze(0).detach().cpu().tolist()
    probabilities = torch.sigmoid(logits).squeeze(0).detach().cpu().tolist()
    predictions = [1 if float(probability) >= _threshold else 0 for probability in probabilities]
    predicted_labels = [
        label for label, prediction in zip(_classes, predictions) if prediction
    ]

    return ScoreResult(
        index=0,
        embedding=[float(value) for value in image_features.squeeze(0).detach().cpu().tolist()],
        scores=[float(value) for value in raw_scores],
        scores_by_class={
            label: float(value) for label, value in zip(_classes, raw_scores)
        },
        probabilities=[float(value) for value in probabilities],
        probabilities_by_class={
            label: float(value) for label, value in zip(_classes, probabilities)
        },
        predictions=[int(value) for value in predictions],
        predictions_by_class={
            label: int(value) for label, value in zip(_classes, predictions)
        },
        predicted_labels=predicted_labels,
        gate_weights=[float(value) for value in gate_weights],
    )


@app.post("/predict", response_model=PredictResponse)
@torch.no_grad()
def predict(request: PredictRequest):
    """
    Classify an ECG image.

    Accepts a base64-encoded image, returns per-class sigmoid scores.
    """
    _ensure_models_loaded()

    start = time.perf_counter()
    image = _decode_base64_image(request.image_base64)
    image_features = _embed_image(image)
    scored = _score_embedding(image_features)
    scores = scored.probabilities

    scores_by_class = {label: score for label, score in zip(_classes, scores)}
    predicted_labels = [
        label for label, score in zip(_classes, scores) if score >= _threshold
    ]

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "predict: predicted=%s top3=%s elapsed_ms=%.1f",
        predicted_labels,
        sorted(scores_by_class.items(), key=lambda x: x[1], reverse=True)[:3],
        elapsed_ms,
    )

    return PredictResponse(
        classifier_type=_classifier_type,
        medsiglip_model_id=_model_id,
        classes=list(_classes),
        scores=scores,
        scores_by_class=scores_by_class,
        predicted_labels=predicted_labels,
        threshold=_threshold,
    )


@app.post("/embed/image", response_model=EmbedImageResponse)
@torch.no_grad()
async def embed_image(file: UploadFile = File(...)):
    """Return the MedSigLIP image embedding for one uploaded file."""
    _ensure_models_loaded()

    raw = await file.read()
    image = _load_image_from_bytes(raw)
    image_features = _embed_image(image)
    embedding = [float(value) for value in image_features.squeeze(0).detach().cpu().tolist()]

    logger.info(
        "embed/image: filename=%s embed_dim=%s",
        file.filename or "",
        len(embedding),
    )

    return EmbedImageResponse(embedding=embedding)


@app.post("/score", response_model=ScoreResponse)
@torch.no_grad()
async def score(file: UploadFile = File(...)):
    """Score one uploaded ECG image via MedSigLIP embedding + classifier head."""
    _ensure_models_loaded()

    start = time.perf_counter()
    raw = await file.read()
    image = _load_image_from_bytes(raw)
    image_features = _embed_image(image)
    result_row = _score_embedding(image_features)

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "score: filename=%s predicted=%s top3=%s elapsed_ms=%.1f",
        file.filename or "",
        result_row.predicted_labels,
        sorted(result_row.probabilities_by_class.items(), key=lambda item: item[1], reverse=True)[:3],
        elapsed_ms,
    )

    return ScoreResponse(
        classifier_type=_classifier_type,
        medsiglip_model_id=_model_id,
        device=_device,
        classes=list(_classes),
        threshold=float(_threshold),
        num_experts=int(_num_experts),
        results=[result_row],
        scoring_mode=(
            "image -> MedSigLIP image embedding -> "
            f"{_classifier_type} logits -> sigmoid probabilities -> thresholded predictions"
        ),
    )


if __name__ == "__main__":
    import uvicorn

    # PORT (generic) → AIP_HTTP_PORT (Vertex AI) → 8080 (default)
    port = int(
        os.environ.get("PORT")
        or os.environ.get("AIP_HTTP_PORT")
        or "8080"
    )
    uvicorn.run(app, host="0.0.0.0", port=port)
