"""
ECG classifier service.

Pipeline:
1) Receive base64-encoded ECG image.
2) Call a remote ECG classifier or MedSigLIP embedding endpoint.
3) Return per-class scores for downstream MedGemma analysis.

Supported remote pipelines:
- image → remote classifier → scores
- image → remote MedSigLIP embedding → local classifier checkpoint → scores

This service only makes an HTTP call; it supports both of the protocols used in this repo:
  - POST /predict with JSON {"image_base64": "..."}
  - POST /score with multipart form-data and prompt texts
  - POST /embed/image with multipart form-data and image file

Supported auth types (via ECG_CLASSIFIER_AUTH_TYPE):
  - none:          No auth headers (local dev, VPN-protected endpoints)
  - bearer:        Static Authorization: Bearer <token>
  - api_key:       Configurable header + key (e.g. X-API-Key)
  - vertex_gcloud: Vertex access token (ADC/service account first, gcloud fallback)
  - vertex:        Alias of vertex_gcloud
"""

from __future__ import annotations

import base64
from functools import lru_cache
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

ECG_LABEL_PROMPTS: list[tuple[str, str]] = [
    ("NORM", "12-lead ECG showing normal sinus rhythm and otherwise normal ECG findings"),
    ("MI", "12-lead ECG showing myocardial infarction pattern or infarction-related changes"),
    ("STTC", "12-lead ECG showing ST segment or T wave change abnormalities"),
    ("CD", "12-lead ECG showing conduction disturbance or bundle branch conduction abnormality"),
    ("HYP", "12-lead ECG showing cardiac chamber hypertrophy or strain pattern"),
]
DEFAULT_SCORE_THRESHOLD = 0.5


class ECGClassifierService:
    """Calls a remote ECG classifier endpoint for predictions.

    Works with remote endpoints that either return per-class scores directly
    or return a MedSigLIP image embedding that this service can score locally.
    Auth is controlled by
    ``ECG_CLASSIFIER_AUTH_TYPE`` (see module docstring).
    """

    def __init__(self) -> None:
        self._timeout = httpx.Timeout(
            float(settings.ecg_classifier_endpoint_timeout),
            connect=30.0,
        )

    def _get_endpoint_url(self) -> str:
        url = (settings.ecg_classifier_endpoint_url or "").strip()
        if not url:
            raise RuntimeError(
                "ECG classifier endpoint is not configured. "
                "Set ECG_CLASSIFIER_ENDPOINT_URL in your environment."
            )
        return url

    def _resolve_request_candidates(self) -> list[tuple[str, str]]:
        """
        Resolve the configured endpoint into concrete request candidates.

        Supported modes:
        - predict_json: POST JSON {"image_base64": "..."} to /predict or Vertex :predict
        - score_multipart: POST multipart image+texts to /score
        - embed_image_multipart: POST multipart file upload to /embed/image
        """
        base_url = self._get_endpoint_url().rstrip("/")
        lowered = base_url.lower()

        if lowered.endswith("/embed/image"):
            return [("embed_image_multipart", base_url)]

        if lowered.endswith("/score"):
            return [("score_multipart", base_url)]

        if lowered.endswith("/predict") or ":predict" in lowered:
            return [("predict_json", base_url)]

        return [
            ("predict_json", f"{base_url}/predict"),
            ("score_multipart", f"{base_url}/score"),
            ("embed_image_multipart", f"{base_url}/embed/image"),
        ]

    def _decode_image_base64(self, image_base64: str) -> bytes:
        payload = (image_base64 or "").strip()
        if not payload:
            raise RuntimeError("ECG image payload is empty.")
        if payload.startswith("data:") and "," in payload:
            payload = payload.split(",", 1)[1]
        try:
            return base64.b64decode(payload)
        except Exception as exc:
            raise RuntimeError(f"Invalid ECG image payload: {exc}") from exc

    def _build_score_request_data(self) -> dict[str, str]:
        return {
            "normalize": "true",
            # The MedSigLIP /score endpoint expects `texts` as a JSON string
            # inside multipart form-data, not as repeated form fields.
            "texts": json.dumps([prompt for _, prompt in ECG_LABEL_PROMPTS]),
        }

    def _resolve_checkpoint_path(self) -> Path:
        configured = str(getattr(settings, "ecg_classifier_checkpoint_path", "") or "").strip()
        if configured:
            return Path(configured).expanduser()
        return (
            Path(__file__).resolve().parents[3]
            / "ecg_classifier"
            / "embed_data"
            / "moe_classifier_medsiglip.pt"
        )

    def _resolve_medsiglip_model_id(self, result: dict[str, Any]) -> str:
        return str(
            result.get("medsiglip_model_id")
            or result.get("model")
            or getattr(settings, "ecg_medsiglip_model_id", "")
            or "google/medsiglip-448"
        )

    def _normalize_score_response(self, result: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(result, dict):
            raise RuntimeError("ECG classifier endpoint returned a non-object response.")

        raw_scores = result.get("scores")
        if not isinstance(raw_scores, list) or not raw_scores:
            raise RuntimeError("ECG classifier endpoint returned missing or empty scores.")

        first_row = raw_scores[0]
        if not isinstance(first_row, list) or not first_row:
            raise RuntimeError("ECG classifier endpoint returned an empty score row.")

        classes = [label for label, _ in ECG_LABEL_PROMPTS]
        if len(first_row) != len(classes):
            raise RuntimeError(
                "ECG classifier endpoint returned wrong score vector length: "
                f"expected={len(classes)} got={len(first_row)}"
            )

        try:
            scores = [float(value) for value in first_row]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "ECG classifier endpoint returned non-numeric scores."
            ) from exc

        scores_by_class = {
            label: score for label, score in zip(classes, scores)
        }
        predicted_labels = [
            label for label, score in zip(classes, scores)
            if score >= DEFAULT_SCORE_THRESHOLD
        ]

        return {
            "classifier_type": "medsiglip_similarity",
            "checkpoint_path": "remote-score-endpoint",
            "medsiglip_model_id": self._resolve_medsiglip_model_id(result),
            "classes": classes,
            "scores": scores,
            "scores_by_class": scores_by_class,
            "predicted_labels": predicted_labels,
            "threshold": DEFAULT_SCORE_THRESHOLD,
        }

    def _normalize_predict_response(self, result: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(result, dict):
            raise RuntimeError("ECG classifier endpoint returned a non-object response.")

        raw_scores = result.get("scores")
        if not isinstance(raw_scores, list) or not raw_scores:
            raise RuntimeError("ECG classifier endpoint returned missing or empty scores.")

        try:
            scores = [float(value) for value in raw_scores]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "ECG classifier endpoint returned non-numeric scores."
            ) from exc

        raw_classes = result.get("classes")
        if isinstance(raw_classes, list) and raw_classes:
            classes = [str(item) for item in raw_classes]
        else:
            classes = [label for label, _ in ECG_LABEL_PROMPTS]

        if len(scores) != len(classes):
            raise RuntimeError(
                "ECG classifier endpoint returned wrong score vector length: "
                f"expected={len(classes)} got={len(scores)}"
            )

        scores_by_class = {
            label: score for label, score in zip(classes, scores)
        }
        predicted_labels = result.get("predicted_labels")
        if isinstance(predicted_labels, list):
            normalized_predicted_labels = [str(item) for item in predicted_labels]
        else:
            threshold = float(result.get("threshold", DEFAULT_SCORE_THRESHOLD))
            normalized_predicted_labels = [
                label for label, score in zip(classes, scores)
                if score >= threshold
            ]

        return {
            "classifier_type": str(result.get("classifier_type") or "remote-predict-endpoint"),
            "checkpoint_path": str(result.get("checkpoint_path") or "remote-predict-endpoint"),
            "medsiglip_model_id": self._resolve_medsiglip_model_id(result),
            "classes": classes,
            "scores": scores,
            "scores_by_class": scores_by_class,
            "predicted_labels": normalized_predicted_labels,
            "threshold": float(result.get("threshold", DEFAULT_SCORE_THRESHOLD)),
        }

    def _normalize_embed_image_response(self, result: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(result, dict):
            raise RuntimeError("MedSigLIP embedding endpoint returned a non-object response.")

        raw_embedding = result.get("embedding")
        if not isinstance(raw_embedding, list) or not raw_embedding:
            raise RuntimeError(
                "MedSigLIP embedding endpoint returned missing or empty embedding."
            )

        try:
            embedding = [float(value) for value in raw_embedding]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "MedSigLIP embedding endpoint returned a non-numeric embedding."
            ) from exc

        return self._predict_from_embedding(
            embedding,
            medsiglip_model_id=self._resolve_medsiglip_model_id(result),
        )

    async def _post_predict_json(
        self,
        client: httpx.AsyncClient,
        endpoint_url: str,
        headers: dict[str, str],
        image_base64: str,
    ) -> httpx.Response:
        return await client.post(
            endpoint_url,
            headers=headers,
            json={"image_base64": image_base64},
        )

    async def _post_score_multipart(
        self,
        client: httpx.AsyncClient,
        endpoint_url: str,
        headers: dict[str, str],
        image_bytes: bytes,
    ) -> httpx.Response:
        multipart_headers = dict(headers)
        multipart_headers.pop("Content-Type", None)
        return await client.post(
            endpoint_url,
            headers=multipart_headers,
            data=self._build_score_request_data(),
            files=[("files", ("ecg-upload.png", image_bytes, "image/png"))],
        )

    async def _post_embed_image_multipart(
        self,
        client: httpx.AsyncClient,
        endpoint_url: str,
        headers: dict[str, str],
        image_bytes: bytes,
    ) -> httpx.Response:
        multipart_headers = dict(headers)
        multipart_headers.pop("Content-Type", None)
        return await client.post(
            endpoint_url,
            headers=multipart_headers,
            files=[("file", ("ecg-upload.png", image_bytes, "image/png"))],
        )

    def _predict_from_embedding(
        self,
        embedding: list[float],
        *,
        medsiglip_model_id: str,
    ) -> dict[str, Any]:
        classifier_state = _load_local_classifier_state(str(self._resolve_checkpoint_path()))

        if len(embedding) != classifier_state["embed_dim"]:
            raise RuntimeError(
                "MedSigLIP embedding endpoint returned wrong embedding length: "
                f"expected={classifier_state['embed_dim']} got={len(embedding)}"
            )

        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "ECG embedding scoring requires PyTorch in the API runtime."
            ) from exc

        with torch.no_grad():
            input_tensor = torch.tensor([embedding], dtype=torch.float32)
            logits_output = classifier_state["model"](input_tensor)
            logits = logits_output[0] if isinstance(logits_output, tuple) else logits_output
            scores = torch.sigmoid(logits).detach().cpu().view(-1).tolist()

        classes = list(classifier_state["classes"])
        scores_by_class = {
            label: float(score)
            for label, score in zip(classes, scores)
        }
        threshold = float(classifier_state["threshold"])
        predicted_labels = [
            label for label, score in zip(classes, scores)
            if score >= threshold
        ]

        return {
            "classifier_type": str(classifier_state["model_type"]),
            "checkpoint_path": f"local-checkpoint:{Path(classifier_state['checkpoint_path']).name}",
            "medsiglip_model_id": medsiglip_model_id,
            "classes": classes,
            "scores": [float(score) for score in scores],
            "scores_by_class": scores_by_class,
            "predicted_labels": predicted_labels,
            "threshold": threshold,
        }

    async def _get_auth_headers(self) -> dict[str, str]:
        """
        Build authentication headers based on the configured auth type.

        Supported types: none, bearer, api_key, vertex_gcloud, vertex.
        """
        auth_type = (settings.ecg_classifier_auth_type or "vertex_gcloud").strip().lower()

        if auth_type == "none":
            return {"Content-Type": "application/json"}

        if auth_type == "bearer":
            token = (settings.ecg_classifier_bearer_token or "").strip()
            if not token:
                raise RuntimeError(
                    "ECG_CLASSIFIER_BEARER_TOKEN is required when "
                    "ECG_CLASSIFIER_AUTH_TYPE=bearer"
                )
            return {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }

        if auth_type == "api_key":
            key = (settings.ecg_classifier_api_key or "").strip()
            header_name = (
                settings.ecg_classifier_api_key_header or "X-API-Key"
            ).strip()
            if not key:
                raise RuntimeError(
                    "ECG_CLASSIFIER_API_KEY is required when "
                    "ECG_CLASSIFIER_AUTH_TYPE=api_key"
                )
            return {
                header_name: key,
                "Content-Type": "application/json",
            }

        if auth_type in {"vertex_gcloud", "vertex"}:
            from app.services.llm_client import llm_client

            token = await llm_client._get_vertex_access_token()
            return {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }

        raise RuntimeError(
            f"Unsupported ECG_CLASSIFIER_AUTH_TYPE='{auth_type}'. "
            "Use: none, bearer, api_key, vertex_gcloud, or vertex."
        )

    async def predict_from_base64(self, image_base64: str) -> dict[str, Any]:
        """
        Call the remote ECG classifier endpoint and adapt the response into
        the historical ECG classifier payload shape.

        The response format matches the previous local implementation so that
        downstream code in llm.py requires no changes.
        """
        start_total = time.perf_counter()
        request_candidates = self._resolve_request_candidates()
        image_bytes = self._decode_image_base64(image_base64)
        logger.info(
            "[ecg-classifier] predict start (remote) endpoint=%s candidates=%s image_base64_len=%s image_bytes_len=%s",
            self._get_endpoint_url(),
            [mode for mode, _ in request_candidates],
            len(image_base64 or ""),
            len(image_bytes),
        )

        try:
            headers = await self._get_auth_headers()
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = None
                last_status_error: httpx.HTTPStatusError | None = None
                selected_mode = ""
                selected_url = ""

                for mode, endpoint_url in request_candidates:
                    selected_mode = mode
                    selected_url = endpoint_url
                    try:
                        if mode == "predict_json":
                            response = await self._post_predict_json(
                                client,
                                endpoint_url,
                                headers,
                                image_base64,
                            )
                        elif mode == "score_multipart":
                            response = await self._post_score_multipart(
                                client,
                                endpoint_url,
                                headers,
                                image_bytes,
                            )
                        elif mode == "embed_image_multipart":
                            response = await self._post_embed_image_multipart(
                                client,
                                endpoint_url,
                                headers,
                                image_bytes,
                            )
                        else:
                            raise RuntimeError(f"Unsupported ECG classifier request mode: {mode}")

                        response.raise_for_status()
                        result = response.json()
                        break
                    except httpx.HTTPStatusError as exc:
                        if (
                            len(request_candidates) > 1
                            and exc.response.status_code in {404, 405, 415, 422}
                        ):
                            last_status_error = exc
                            logger.warning(
                                "[ecg-classifier] request mode failed; trying fallback mode=%s url=%s status=%s",
                                mode,
                                endpoint_url,
                                exc.response.status_code,
                            )
                            continue
                        raise
                else:
                    if last_status_error is not None:
                        raise last_status_error
                    raise RuntimeError("ECG classifier endpoint returned no response.")

        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to ECG classifier endpoint: {selected_url or self._get_endpoint_url()}"
            )
        except httpx.TimeoutException:
            raise RuntimeError(
                "ECG classifier endpoint request timed out "
                f"after {float(settings.ecg_classifier_endpoint_timeout):.0f}s "
                f"(mode={selected_mode or request_candidates[0][0]} "
                f"url={selected_url or request_candidates[0][1]})"
            )
        except httpx.HTTPStatusError as exc:
            error_detail = ""
            try:
                error_detail = exc.response.text[:500]
            except Exception:
                pass
            raise RuntimeError(
                f"ECG classifier endpoint error ({exc.response.status_code}): {error_detail}"
            )

        if selected_mode == "predict_json":
            prediction = self._normalize_predict_response(result)
        elif selected_mode == "embed_image_multipart":
            prediction = self._normalize_embed_image_response(result)
        else:
            prediction = self._normalize_score_response(result)

        elapsed_ms = (time.perf_counter() - start_total) * 1000
        logger.info(
            "[ecg-classifier] predict complete (remote) mode=%s url=%s classifier_type=%s predicted=%s "
            "top3=%s elapsed_ms=%.1f",
            selected_mode,
            selected_url,
            prediction["classifier_type"],
            prediction["predicted_labels"],
            sorted(prediction["scores_by_class"].items(), key=lambda x: x[1], reverse=True)[:3],
            elapsed_ms,
        )

        return prediction


ecg_classifier_service = ECGClassifierService()


@lru_cache(maxsize=4)
def _load_local_classifier_state(checkpoint_path: str) -> dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "ECG embedding scoring requires PyTorch in the API runtime."
        ) from exc

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

        def forward(self, x: Any) -> Any:
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

        def forward(self, x: Any) -> Any:
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

        def forward(self, x: Any) -> Any:
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.out(x)

    def build_classifier(ckpt: dict[str, Any]) -> tuple[Any, str]:
        state_dict = ckpt.get("state_dict")
        if not isinstance(state_dict, dict) or not state_dict:
            raise RuntimeError("Checkpoint missing state_dict.")

        embed_dim = int(ckpt["embed_dim"])
        num_classes = int(ckpt["num_classes"])

        if any(key.startswith("experts.") for key in state_dict):
            num_experts = int(ckpt.get("num_experts", 5))
            expert_linear_layers: list[tuple[int, Any]] = []
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

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"ECG classifier checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model, model_type = build_classifier(ckpt)
    classes = [str(item) for item in (ckpt.get("classes") or [])]
    if not classes:
        classes = [label for label, _ in ECG_LABEL_PROMPTS]

    configured_threshold = getattr(settings, "ecg_classifier_threshold", DEFAULT_SCORE_THRESHOLD)
    threshold = float(
        configured_threshold
        if configured_threshold is not None
        else ckpt.get("threshold", DEFAULT_SCORE_THRESHOLD)
    )
    embed_dim = int(ckpt.get("embed_dim") or 0)
    if embed_dim <= 0:
        raise RuntimeError("Checkpoint missing valid embed_dim.")

    return {
        "model": model,
        "model_type": model_type,
        "classes": classes,
        "threshold": threshold,
        "embed_dim": embed_dim,
        "checkpoint_path": str(path),
    }
