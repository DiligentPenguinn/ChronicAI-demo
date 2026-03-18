"""
ECG classifier service.

Pipeline:
1) Receive base64-encoded ECG image.
2) Call a remote ECG classifier endpoint.
3) Return per-class scores for downstream MedGemma analysis.

Supported remote pipelines:
- image -> remote classifier -> scores

This service only makes an HTTP call; it supports both remote scoring protocols used in this repo:
  - POST /predict with JSON {"image_base64": "..."}
  - POST /score with multipart form-data and one image file

Supported auth types (via ECG_CLASSIFIER_AUTH_TYPE):
  - none:          No auth headers (local dev, VPN-protected endpoints)
  - bearer:        Static Authorization: Bearer <token>
  - api_key:       Configurable header + key (e.g. X-API-Key)
  - vertex_gcloud: Vertex access token (ADC/service account first, gcloud fallback)
  - vertex:        Alias of vertex_gcloud
"""

from __future__ import annotations

import base64
import logging
import math
import time
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

    Works with remote endpoints that return per-class scores directly.
    Auth is controlled by ``ECG_CLASSIFIER_AUTH_TYPE`` (see module docstring).
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
        - score_multipart: POST multipart image file to /score
        """
        base_url = self._get_endpoint_url().rstrip("/")
        lowered = base_url.lower()

        if lowered.endswith("/score"):
            return [("score_multipart", base_url)]

        if lowered.endswith("/predict") or ":predict" in lowered:
            return [("predict_json", base_url)]

        return [
            ("predict_json", f"{base_url}/predict"),
            ("score_multipart", f"{base_url}/score"),
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

    def _resolve_medsiglip_model_id(self, result: dict[str, Any]) -> str:
        return str(
            result.get("medsiglip_model_id")
            or result.get("model")
            or getattr(settings, "ecg_medsiglip_model_id", "")
            or "google/medsiglip-448"
        )

    def _default_classes(self) -> list[str]:
        return [label for label, _ in ECG_LABEL_PROMPTS]

    def _resolve_classes(self, result: dict[str, Any], row: dict[str, Any] | None = None) -> list[str]:
        for candidate in (result.get("classes"), (row or {}).get("classes")):
            if isinstance(candidate, list) and candidate:
                return [str(item) for item in candidate]

        for key in ("scores_by_class", "probabilities_by_class", "predictions_by_class"):
            candidate = (row or {}).get(key)
            if isinstance(candidate, dict) and candidate:
                return [str(label) for label in candidate.keys()]

        return self._default_classes()

    def _coerce_float_list(
        self,
        raw_values: Any,
        *,
        field_name: str,
        allow_empty: bool = False,
    ) -> list[float]:
        if raw_values is None:
            if allow_empty:
                return []
            raise RuntimeError(f"ECG classifier endpoint returned missing {field_name}.")
        if not isinstance(raw_values, list):
            raise RuntimeError(f"ECG classifier endpoint returned non-list {field_name}.")
        if not raw_values and not allow_empty:
            raise RuntimeError(f"ECG classifier endpoint returned empty {field_name}.")
        try:
            return [float(value) for value in raw_values]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"ECG classifier endpoint returned non-numeric {field_name}."
            ) from exc

    def _coerce_binary_list(
        self,
        raw_values: Any,
        *,
        field_name: str,
        allow_empty: bool = False,
    ) -> list[int]:
        if raw_values is None:
            if allow_empty:
                return []
            raise RuntimeError(f"ECG classifier endpoint returned missing {field_name}.")
        if not isinstance(raw_values, list):
            raise RuntimeError(f"ECG classifier endpoint returned non-list {field_name}.")
        if not raw_values and not allow_empty:
            raise RuntimeError(f"ECG classifier endpoint returned empty {field_name}.")

        normalized: list[int] = []
        for value in raw_values:
            try:
                int_value = int(value)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"ECG classifier endpoint returned non-binary {field_name}."
                ) from exc
            if int_value not in {0, 1}:
                raise RuntimeError(
                    f"ECG classifier endpoint returned non-binary {field_name}."
                )
            normalized.append(int_value)
        return normalized

    def _build_float_mapping(
        self,
        raw_mapping: Any,
        *,
        classes: list[str],
        fallback_values: list[float],
    ) -> dict[str, float]:
        if isinstance(raw_mapping, dict) and raw_mapping:
            try:
                return {
                    label: float(raw_mapping[label])
                    for label in classes
                    if label in raw_mapping
                }
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "ECG classifier endpoint returned non-numeric per-class values."
                ) from exc
        return {
            label: float(value)
            for label, value in zip(classes, fallback_values)
        }

    def _build_binary_mapping(
        self,
        raw_mapping: Any,
        *,
        classes: list[str],
        fallback_values: list[int],
    ) -> dict[str, int]:
        if isinstance(raw_mapping, dict) and raw_mapping:
            normalized: dict[str, int] = {}
            for label in classes:
                if label not in raw_mapping:
                    continue
                try:
                    int_value = int(raw_mapping[label])
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "ECG classifier endpoint returned non-binary per-class predictions."
                    ) from exc
                if int_value not in {0, 1}:
                    raise RuntimeError(
                        "ECG classifier endpoint returned non-binary per-class predictions."
                    )
                normalized[label] = int_value
            if normalized:
                return normalized
        return {
            label: int(value)
            for label, value in zip(classes, fallback_values)
        }

    def _validate_vector_length(
        self,
        values: list[Any],
        *,
        classes: list[str],
        field_name: str,
    ) -> None:
        if len(values) != len(classes):
            raise RuntimeError(
                "ECG classifier endpoint returned wrong score vector length: "
                f"expected={len(classes)} got={len(values)} ({field_name})"
            )

    def _sigmoid(self, value: float) -> float:
        if value >= 0:
            exp_value = math.exp(-value)
            return 1.0 / (1.0 + exp_value)
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    def _resolve_probabilities(self, raw_probabilities: Any, *, scores: list[float]) -> list[float]:
        if isinstance(raw_probabilities, list) and raw_probabilities:
            probabilities = self._coerce_float_list(
                raw_probabilities,
                field_name="probabilities",
            )
            if any(probability < 0.0 or probability > 1.0 for probability in probabilities):
                raise RuntimeError(
                    "ECG classifier endpoint returned probabilities outside [0, 1]."
                )
            return probabilities

        if scores and all(0.0 <= score <= 1.0 for score in scores):
            return [float(score) for score in scores]
        return [self._sigmoid(score) for score in scores]

    def _normalize_moe_score_response(self, result: dict[str, Any]) -> dict[str, Any]:
        rows = result.get("results")
        if not isinstance(rows, list) or not rows:
            raise RuntimeError("ECG classifier endpoint returned missing or empty results.")

        first_row = rows[0]
        if not isinstance(first_row, dict):
            raise RuntimeError("ECG classifier endpoint returned a non-object result row.")

        classes = self._resolve_classes(result, first_row)
        scores = self._coerce_float_list(first_row.get("scores"), field_name="scores")
        self._validate_vector_length(scores, classes=classes, field_name="scores")

        probabilities = self._resolve_probabilities(
            first_row.get("probabilities"),
            scores=scores,
        )
        self._validate_vector_length(
            probabilities,
            classes=classes,
            field_name="probabilities",
        )

        threshold = float(result.get("threshold", first_row.get("threshold", DEFAULT_SCORE_THRESHOLD)))
        raw_predictions = first_row.get("predictions")
        if isinstance(raw_predictions, list) and raw_predictions:
            predictions = self._coerce_binary_list(raw_predictions, field_name="predictions")
        else:
            predictions = [1 if probability >= threshold else 0 for probability in probabilities]
        self._validate_vector_length(predictions, classes=classes, field_name="predictions")

        raw_predicted_labels = first_row.get("predicted_labels")
        if isinstance(raw_predicted_labels, list):
            predicted_labels = [str(item) for item in raw_predicted_labels if str(item).strip()]
        else:
            predicted_labels = [
                label for label, predicted in zip(classes, predictions)
                if predicted
            ]

        return {
            "classifier_type": str(result.get("classifier_type") or "moe_classifier"),
            "checkpoint_path": str(result.get("checkpoint_path") or "remote-score-endpoint"),
            "medsiglip_model_id": self._resolve_medsiglip_model_id(result),
            "device": str(result.get("device") or ""),
            "classes": classes,
            "scores": scores,
            "scores_by_class": self._build_float_mapping(
                first_row.get("scores_by_class"),
                classes=classes,
                fallback_values=scores,
            ),
            "probabilities": probabilities,
            "probabilities_by_class": self._build_float_mapping(
                first_row.get("probabilities_by_class"),
                classes=classes,
                fallback_values=probabilities,
            ),
            "predictions": predictions,
            "predictions_by_class": self._build_binary_mapping(
                first_row.get("predictions_by_class"),
                classes=classes,
                fallback_values=predictions,
            ),
            "predicted_labels": predicted_labels,
            "threshold": threshold,
            "embedding": self._coerce_float_list(
                first_row.get("embedding"),
                field_name="embedding",
                allow_empty=True,
            ),
            "gate_weights": self._coerce_float_list(
                first_row.get("gate_weights"),
                field_name="gate_weights",
                allow_empty=True,
            ),
            "num_experts": int(result.get("num_experts") or 0),
            "scoring_mode": str(result.get("scoring_mode") or ""),
        }

    def _normalize_legacy_score_response(self, result: dict[str, Any]) -> dict[str, Any]:
        raw_scores = result.get("scores")
        if not isinstance(raw_scores, list) or not raw_scores:
            raise RuntimeError("ECG classifier endpoint returned missing or empty scores.")

        first_row = raw_scores[0]
        if not isinstance(first_row, list) or not first_row:
            raise RuntimeError("ECG classifier endpoint returned an empty score row.")

        classes = self._default_classes()
        scores = self._coerce_float_list(first_row, field_name="scores")
        self._validate_vector_length(scores, classes=classes, field_name="scores")
        probabilities = self._resolve_probabilities(result.get("probabilities"), scores=scores)
        self._validate_vector_length(
            probabilities,
            classes=classes,
            field_name="probabilities",
        )

        threshold = float(result.get("threshold", DEFAULT_SCORE_THRESHOLD))
        predictions = [1 if probability >= threshold else 0 for probability in probabilities]
        predicted_labels = [
            label for label, predicted in zip(classes, predictions)
            if predicted
        ]

        return {
            "classifier_type": str(result.get("classifier_type") or "remote-score-endpoint"),
            "checkpoint_path": str(result.get("checkpoint_path") or "remote-score-endpoint"),
            "medsiglip_model_id": self._resolve_medsiglip_model_id(result),
            "device": str(result.get("device") or ""),
            "classes": classes,
            "scores": scores,
            "scores_by_class": self._build_float_mapping(
                result.get("scores_by_class"),
                classes=classes,
                fallback_values=scores,
            ),
            "probabilities": probabilities,
            "probabilities_by_class": self._build_float_mapping(
                result.get("probabilities_by_class"),
                classes=classes,
                fallback_values=probabilities,
            ),
            "predictions": predictions,
            "predictions_by_class": self._build_binary_mapping(
                result.get("predictions_by_class"),
                classes=classes,
                fallback_values=predictions,
            ),
            "predicted_labels": predicted_labels,
            "threshold": threshold,
            "embedding": [],
            "gate_weights": [],
            "num_experts": int(result.get("num_experts") or 0),
            "scoring_mode": str(result.get("scoring_mode") or ""),
        }

    def _normalize_score_response(self, result: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(result, dict):
            raise RuntimeError("ECG classifier endpoint returned a non-object response.")
        if isinstance(result.get("results"), list):
            return self._normalize_moe_score_response(result)
        return self._normalize_legacy_score_response(result)

    def _normalize_predict_response(self, result: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(result, dict):
            raise RuntimeError("ECG classifier endpoint returned a non-object response.")

        scores = self._coerce_float_list(result.get("scores"), field_name="scores")
        classes = self._resolve_classes(result)
        self._validate_vector_length(scores, classes=classes, field_name="scores")

        probabilities = self._resolve_probabilities(result.get("probabilities"), scores=scores)
        self._validate_vector_length(
            probabilities,
            classes=classes,
            field_name="probabilities",
        )

        threshold = float(result.get("threshold", DEFAULT_SCORE_THRESHOLD))
        raw_predictions = result.get("predictions")
        if isinstance(raw_predictions, list) and raw_predictions:
            predictions = self._coerce_binary_list(raw_predictions, field_name="predictions")
        else:
            predictions = [1 if probability >= threshold else 0 for probability in probabilities]
        self._validate_vector_length(predictions, classes=classes, field_name="predictions")

        predicted_labels = result.get("predicted_labels")
        if isinstance(predicted_labels, list):
            normalized_predicted_labels = [str(item) for item in predicted_labels]
        else:
            normalized_predicted_labels = [
                label for label, predicted in zip(classes, predictions)
                if predicted
            ]

        return {
            "classifier_type": str(result.get("classifier_type") or "remote-predict-endpoint"),
            "checkpoint_path": str(result.get("checkpoint_path") or "remote-predict-endpoint"),
            "medsiglip_model_id": self._resolve_medsiglip_model_id(result),
            "device": str(result.get("device") or ""),
            "classes": classes,
            "scores": scores,
            "scores_by_class": self._build_float_mapping(
                result.get("scores_by_class"),
                classes=classes,
                fallback_values=scores,
            ),
            "probabilities": probabilities,
            "probabilities_by_class": self._build_float_mapping(
                result.get("probabilities_by_class"),
                classes=classes,
                fallback_values=probabilities,
            ),
            "predictions": predictions,
            "predictions_by_class": self._build_binary_mapping(
                result.get("predictions_by_class"),
                classes=classes,
                fallback_values=predictions,
            ),
            "predicted_labels": normalized_predicted_labels,
            "threshold": threshold,
            "embedding": self._coerce_float_list(
                result.get("embedding"),
                field_name="embedding",
                allow_empty=True,
            ),
            "gate_weights": self._coerce_float_list(
                result.get("gate_weights"),
                field_name="gate_weights",
                allow_empty=True,
            ),
            "num_experts": int(result.get("num_experts") or 0),
            "scoring_mode": str(result.get("scoring_mode") or ""),
        }

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
            files=[("file", ("ecg-upload.png", image_bytes, "image/png"))],
        )

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
