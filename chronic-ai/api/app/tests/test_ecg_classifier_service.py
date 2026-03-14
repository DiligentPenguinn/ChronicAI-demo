"""
Unit tests for ECGClassifierService auth dispatcher.

Tests verify that _get_auth_headers produces correct headers for each
ECG_CLASSIFIER_AUTH_TYPE without touching the network or gcloud CLI.
"""
import base64
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Stub out heavy dependencies that app.services.__init__ eagerly imports.
# This lets us test ecg_classifier_service in isolation.
# ---------------------------------------------------------------------------
_STUBS = [
    "langchain_core", "langchain_core.runnables",
    "langgraph", "langgraph.graph", "langgraph.graph.state",
    "langgraph.types", "langgraph.checkpoint", "langgraph.checkpoint.memory",
]
for mod_name in _STUBS:
    sys.modules.setdefault(mod_name, MagicMock())

# Now it's safe to import
from app.services.ecg_classifier_service import (  # noqa: E402
    DEFAULT_SCORE_THRESHOLD,
    ECG_LABEL_PROMPTS,
    ECGClassifierService,
)


@pytest.fixture
def service():
    """Create a fresh ECGClassifierService instance."""
    return ECGClassifierService()


class TestAuthHeaders:
    """Test _get_auth_headers for each auth type."""

    @pytest.mark.asyncio
    async def test_auth_type_none(self, service):
        with patch("app.services.ecg_classifier_service.settings") as ms:
            ms.ecg_classifier_auth_type = "none"
            headers = await service._get_auth_headers()

        assert headers == {"Content-Type": "application/json"}
        assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_auth_type_bearer(self, service):
        with patch("app.services.ecg_classifier_service.settings") as ms:
            ms.ecg_classifier_auth_type = "bearer"
            ms.ecg_classifier_bearer_token = "my-secret-token"
            headers = await service._get_auth_headers()

        assert headers["Authorization"] == "Bearer my-secret-token"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_auth_type_bearer_missing_token(self, service):
        with patch("app.services.ecg_classifier_service.settings") as ms:
            ms.ecg_classifier_auth_type = "bearer"
            ms.ecg_classifier_bearer_token = ""
            with pytest.raises(RuntimeError, match="ECG_CLASSIFIER_BEARER_TOKEN"):
                await service._get_auth_headers()

    @pytest.mark.asyncio
    async def test_auth_type_api_key_default_header(self, service):
        with patch("app.services.ecg_classifier_service.settings") as ms:
            ms.ecg_classifier_auth_type = "api_key"
            ms.ecg_classifier_api_key = "key-12345"
            ms.ecg_classifier_api_key_header = "X-API-Key"
            headers = await service._get_auth_headers()

        assert headers["X-API-Key"] == "key-12345"
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_auth_type_api_key_custom_header(self, service):
        with patch("app.services.ecg_classifier_service.settings") as ms:
            ms.ecg_classifier_auth_type = "api_key"
            ms.ecg_classifier_api_key = "key-12345"
            ms.ecg_classifier_api_key_header = "X-Custom-Auth"
            headers = await service._get_auth_headers()

        assert headers["X-Custom-Auth"] == "key-12345"

    @pytest.mark.asyncio
    async def test_auth_type_api_key_missing_key(self, service):
        with patch("app.services.ecg_classifier_service.settings") as ms:
            ms.ecg_classifier_auth_type = "api_key"
            ms.ecg_classifier_api_key = ""
            ms.ecg_classifier_api_key_header = "X-API-Key"
            with pytest.raises(RuntimeError, match="ECG_CLASSIFIER_API_KEY"):
                await service._get_auth_headers()

    @pytest.mark.asyncio
    async def test_auth_type_vertex_gcloud(self, service):
        mock_llm_client = AsyncMock()
        mock_llm_client._get_vertex_access_token = AsyncMock(
            return_value="gcloud-token-xyz"
        )

        # app.services.__init__ re-exports `llm_client` (the instance)
        # as the module-level `app.services.llm_client`, overriding the module.
        # We need to temporarily put a mock module with the expected attribute.
        mock_module = MagicMock()
        mock_module.llm_client = mock_llm_client

        with patch("app.services.ecg_classifier_service.settings") as ms:
            ms.ecg_classifier_auth_type = "vertex_gcloud"
            with patch.dict(sys.modules, {"app.services.llm_client": mock_module}):
                headers = await service._get_auth_headers()

        assert headers["Authorization"] == "Bearer gcloud-token-xyz"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_auth_type_unsupported(self, service):
        with patch("app.services.ecg_classifier_service.settings") as ms:
            ms.ecg_classifier_auth_type = "oauth2"
            with pytest.raises(RuntimeError, match="Unsupported"):
                await service._get_auth_headers()


class _MockResponse:
    def __init__(self, payload, status_code=200, text="", request_url="https://example.test/score"):
        self._payload = payload
        self.status_code = status_code
        self.text = text
        self.request = httpx.Request("POST", request_url)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "request failed",
                request=self.request,
                response=httpx.Response(
                    self.status_code,
                    request=self.request,
                    text=self.text,
                ),
            )

    def json(self):
        return self._payload


class _MockAsyncClient:
    def __init__(self, response_payload, recorder):
        self._response_payload = response_payload
        self._recorder = recorder

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, **kwargs):
        self._recorder.setdefault("calls", []).append({"url": url, "kwargs": kwargs})
        self._recorder["url"] = url
        self._recorder["kwargs"] = kwargs

        if callable(self._response_payload):
            response = self._response_payload(url, kwargs)
        else:
            response = self._response_payload

        if isinstance(response, _MockResponse):
            return response
        return _MockResponse(response, request_url=url)


class TestPredictFromBase64:
    @pytest.mark.asyncio
    async def test_predict_from_base64_adapts_remote_score_payload(self, service):
        recorder = {}
        image_bytes = b"fake-png-data"
        image_base64 = base64.b64encode(image_bytes).decode("ascii")
        remote_payload = {
            "model": "google/medsiglip-448",
            "device": "cpu",
            "scores": [[0.91, 0.22, 0.61, 0.49, 0.85]],
            "normalized": True,
        }

        with patch("app.services.ecg_classifier_service.settings") as ms:
            ms.ecg_classifier_endpoint_url = "https://example.test/api/score"
            with patch.object(service, "_get_auth_headers", AsyncMock(return_value={"Authorization": "Bearer token", "Content-Type": "application/json"})):
                with patch(
                    "app.services.ecg_classifier_service.httpx.AsyncClient",
                    return_value=_MockAsyncClient(remote_payload, recorder),
                ):
                    result = await service.predict_from_base64(image_base64)

        expected_classes = [label for label, _ in ECG_LABEL_PROMPTS]
        assert recorder["url"] == "https://example.test/api/score"
        assert recorder["kwargs"]["headers"] == {"Authorization": "Bearer token"}
        assert recorder["kwargs"]["data"]["normalize"] == "true"
        assert recorder["kwargs"]["data"]["texts"] == [
            prompt for _, prompt in ECG_LABEL_PROMPTS
        ]
        uploaded_file = recorder["kwargs"]["files"][0]
        assert uploaded_file[0] == "files"
        assert uploaded_file[1][0] == "ecg-upload.png"
        assert uploaded_file[1][1] == image_bytes
        assert result["classifier_type"] == "medsiglip_similarity"
        assert result["checkpoint_path"] == "remote-score-endpoint"
        assert result["medsiglip_model_id"] == "google/medsiglip-448"
        assert result["classes"] == expected_classes
        assert result["scores"] == remote_payload["scores"][0]
        assert result["scores_by_class"]["NORM"] == 0.91
        assert result["scores_by_class"]["HYP"] == 0.85
        assert result["predicted_labels"] == ["NORM", "STTC", "HYP"]
        assert result["threshold"] == DEFAULT_SCORE_THRESHOLD

    @pytest.mark.asyncio
    async def test_predict_from_base64_keeps_explicit_score_endpoint(self, service):
        recorder = {}
        image_base64 = base64.b64encode(b"img").decode("ascii")
        remote_payload = {"model": "m", "scores": [[0.1, 0.2, 0.3, 0.4, 0.5]]}

        with patch("app.services.ecg_classifier_service.settings") as ms:
            ms.ecg_classifier_endpoint_url = "https://example.test/custom/score"
            with patch.object(service, "_get_auth_headers", AsyncMock(return_value={})):
                with patch(
                    "app.services.ecg_classifier_service.httpx.AsyncClient",
                    return_value=_MockAsyncClient(remote_payload, recorder),
                ):
                    await service.predict_from_base64(image_base64)

        assert recorder["url"] == "https://example.test/custom/score"

    @pytest.mark.asyncio
    async def test_predict_from_base64_supports_explicit_predict_endpoint(self, service):
        recorder = {}
        image_base64 = base64.b64encode(b"img").decode("ascii")
        remote_payload = {
            "classifier_type": "moe",
            "checkpoint_path": "vertex-endpoint",
            "medsiglip_model_id": "google/medsiglip-448",
            "classes": [label for label, _ in ECG_LABEL_PROMPTS],
            "scores": [0.1, 0.2, 0.8, 0.4, 0.5],
            "predicted_labels": ["STTC", "HYP"],
            "threshold": 0.5,
        }

        with patch("app.services.ecg_classifier_service.settings") as ms:
            ms.ecg_classifier_endpoint_url = "https://example.test/predict"
            with patch.object(service, "_get_auth_headers", AsyncMock(return_value={"Authorization": "Bearer token", "Content-Type": "application/json"})):
                with patch(
                    "app.services.ecg_classifier_service.httpx.AsyncClient",
                    return_value=_MockAsyncClient(remote_payload, recorder),
                ):
                    result = await service.predict_from_base64(image_base64)

        assert recorder["url"] == "https://example.test/predict"
        assert recorder["kwargs"]["headers"]["Content-Type"] == "application/json"
        assert recorder["kwargs"]["json"] == {"image_base64": image_base64}
        assert "files" not in recorder["kwargs"]
        assert result["classifier_type"] == "moe"
        assert result["checkpoint_path"] == "vertex-endpoint"
        assert result["scores"] == remote_payload["scores"]
        assert result["predicted_labels"] == ["STTC", "HYP"]

    @pytest.mark.asyncio
    async def test_predict_from_base64_tries_predict_then_falls_back_to_score(self, service):
        recorder = {}
        image_bytes = b"fake-png-data"
        image_base64 = base64.b64encode(image_bytes).decode("ascii")
        remote_payload = {
            "model": "google/medsiglip-448",
            "scores": [[0.91, 0.22, 0.61, 0.49, 0.85]],
        }

        def responder(url, kwargs):
            if url.endswith("/predict"):
                return _MockResponse(
                    {"detail": "not found"},
                    status_code=404,
                    text="not found",
                    request_url=url,
                )
            return _MockResponse(remote_payload, request_url=url)

        with patch("app.services.ecg_classifier_service.settings") as ms:
            ms.ecg_classifier_endpoint_url = "https://example.test/base"
            with patch.object(service, "_get_auth_headers", AsyncMock(return_value={"Authorization": "Bearer token", "Content-Type": "application/json"})):
                with patch(
                    "app.services.ecg_classifier_service.httpx.AsyncClient",
                    return_value=_MockAsyncClient(responder, recorder),
                ):
                    result = await service.predict_from_base64(image_base64)

        assert [call["url"] for call in recorder["calls"]] == [
            "https://example.test/base/predict",
            "https://example.test/base/score",
        ]
        assert recorder["calls"][0]["kwargs"]["json"] == {"image_base64": image_base64}
        uploaded_file = recorder["calls"][1]["kwargs"]["files"][0]
        assert uploaded_file[1][1] == image_bytes
        assert result["predicted_labels"] == ["NORM", "STTC", "HYP"]

    @pytest.mark.asyncio
    async def test_predict_from_base64_rejects_wrong_score_vector_length(self, service):
        image_base64 = base64.b64encode(b"img").decode("ascii")
        remote_payload = {"model": "m", "scores": [[0.1, 0.2]]}

        with patch("app.services.ecg_classifier_service.settings") as ms:
            ms.ecg_classifier_endpoint_url = "https://example.test/score"
            with patch.object(service, "_get_auth_headers", AsyncMock(return_value={})):
                with patch(
                    "app.services.ecg_classifier_service.httpx.AsyncClient",
                    return_value=_MockAsyncClient(remote_payload, {}),
                ):
                    with pytest.raises(RuntimeError, match="wrong score vector length"):
                        await service.predict_from_base64(image_base64)

    @pytest.mark.asyncio
    async def test_predict_from_base64_rejects_non_numeric_scores(self, service):
        image_base64 = base64.b64encode(b"img").decode("ascii")
        remote_payload = {"model": "m", "scores": [["bad", 0.2, 0.3, 0.4, 0.5]]}

        with patch("app.services.ecg_classifier_service.settings") as ms:
            ms.ecg_classifier_endpoint_url = "https://example.test/score"
            with patch.object(service, "_get_auth_headers", AsyncMock(return_value={})):
                with patch(
                    "app.services.ecg_classifier_service.httpx.AsyncClient",
                    return_value=_MockAsyncClient(remote_payload, {}),
                ):
                    with pytest.raises(RuntimeError, match="non-numeric scores"):
                        await service.predict_from_base64(image_base64)

    def test_build_score_request_data_is_async_safe_for_httpx_multipart(self, service):
        request = httpx.AsyncClient().build_request(
            "POST",
            "https://example.test/score",
            data=service._build_score_request_data(),
            files={"files": ("ecg-upload.png", b"img", "image/png")},
        )

        assert isinstance(request.stream, httpx._multipart.MultipartStream)
