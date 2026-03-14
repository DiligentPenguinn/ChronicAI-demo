"""
ECG classifier service.

Pipeline:
1) Receive base64-encoded ECG image.
2) Call a remote ECG classifier endpoint (MedSigLIP + MoE).
3) Return per-class scores for downstream MedGemma analysis.

The endpoint runs the full pipeline: image → MedSigLIP embedding → classifier → scores.
This service only makes an HTTP call; it supports both of the protocols used in this repo:
  - POST /predict with JSON {"image_base64": "..."}
  - POST /score with multipart form-data and prompt texts

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

    Works with any HTTP endpoint that accepts ``{"image_base64": "..."}``
    and returns per-class scores.  Auth is controlled by
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

    def _build_score_request_data(self) -> dict[str, str | list[str]]:
        return {
            "normalize": "true",
            "texts": [prompt for _, prompt in ECG_LABEL_PROMPTS],
        }

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
            "medsiglip_model_id": str(result.get("model") or ""),
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
            "medsiglip_model_id": str(
                result.get("medsiglip_model_id") or result.get("model") or ""
            ),
            "classes": classes,
            "scores": scores,
            "scores_by_class": scores_by_class,
            "predicted_labels": normalized_predicted_labels,
            "threshold": float(result.get("threshold", DEFAULT_SCORE_THRESHOLD)),
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
            data=self._build_score_request_data(),
            files=[("files", ("ecg-upload.png", image_bytes, "image/png"))],
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
