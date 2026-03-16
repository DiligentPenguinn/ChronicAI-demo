"""
Unit tests for LLM (Translation Sandwich) Pipeline.
"""
import pytest
from pydantic import ValidationError


class TestTranslationSandwich:
    """Tests for translation sandwich pipeline - requires Ollama running."""
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires translation models")
    async def test_vi_to_en_translation(self):
        """Vietnamese to English translation works."""
        from app.services.llm import translate_vi_to_en
        
        result = await translate_vi_to_en("Xin chào, tôi đau đầu.")
        
        # Should contain some English
        assert any(word in result.lower() for word in ["hello", "headache", "head", "pain"])
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires translation models")
    async def test_en_to_vi_translation(self):
        """English to Vietnamese translation works."""
        from app.services.llm import translate_en_to_vi
        
        result = await translate_en_to_vi("Hello, I have a headache.")
        
        # Should contain Vietnamese characters
        assert any(char in result for char in "àáâãèéêìíòóôõùúýăđĩũơưạ")
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires Ollama with all models")
    async def test_full_pipeline_yields_stages(self):
        """Process medical query yields progress stages."""
        from app.services.llm import process_medical_query
        from uuid import uuid4
        
        stages = []
        async for update in process_medical_query(
            user_input_vi="Tôi bị đau đầu",
            patient_id=uuid4(),
            image_path=None
        ):
            stages.append(update.get("stage"))
        
        # Should have all pipeline stages
        assert "translating_input" in stages
        assert "medical_reasoning" in stages
        assert "translating_output" in stages
        assert "complete" in stages


class TestSystemHealth:
    """Tests for system health check."""
    
    @pytest.mark.asyncio
    async def test_health_check_format(self):
        """Health check returns expected format."""
        from app.services.llm import check_system_health
        
        result = await check_system_health()
        
        assert "status" in result
        assert "provider" in result
        assert "llm" in result
        assert result["status"] in ["healthy", "degraded", "unhealthy"]


class TestPatientSummaryFormatting:
    """Regression tests for patient summary markdown normalization."""

    def test_normalize_patient_summary_sections_and_numbered_lists(self):
        from app.services.llm import _normalize_patient_summary_markdown

        raw = (
            "**Danh sách vấn đề (Problem List)**1.[I10] Tăng huyết áp — Giai đoạn 2, đang điều trị."
            "2.[E78.5] Rối loạn lipid máu — LDL cao, đang điều trị."
            "**Thuốc đang dùng (Current Medications)**1.Losartan 50mg, uống 1 lần/ngày vào buổi sáng."
            "2.Atorvastatin 20mg, uống 1 lần/ngày vào buổi tối."
            "**Dị ứng (Allergies)**NSAIDs"
        )

        formatted = _normalize_patient_summary_markdown(raw)

        assert "## Danh sách vấn đề (Problem List)" in formatted
        assert "## Thuốc đang dùng (Current Medications)" in formatted
        assert "## Dị ứng (Allergies)" in formatted
        assert "\n1. [I10]" in formatted
        assert "\n2. [E78.5]" in formatted
        assert "\n1. Losartan 50mg" in formatted
        assert "\n2. Atorvastatin 20mg" in formatted

    def test_normalize_patient_summary_removes_unbalanced_bold(self):
        from app.services.llm import _normalize_patient_summary_markdown

        raw = (
            "**Diễn tiến bệnh (Disease Progress)Bệnh nhân có tiền sử tăng huyết áp và rối loạn lipid máu."
            "Huyết áp gần nhất 120/80 mmHg."
        )

        formatted = _normalize_patient_summary_markdown(raw)

        assert "**" not in formatted
        assert "## Diễn tiến bệnh (Disease Progress)" in formatted

    def test_normalize_patient_summary_does_not_promote_inline_allergy_phrase(self):
        from app.services.llm import _normalize_patient_summary_markdown

        raw = (
            "## Dị ứng (Allergies)\n"
            "NSAIDs.\n\n"
            "## Đánh giá lâm sàng (Clinical Assessment)\n"
            "Lưu ý tương tác thuốc giảm đau (tránh NSAIDs do Dị ứng (Allergies))."
        )

        formatted = _normalize_patient_summary_markdown(raw)

        assert formatted.count("## Dị ứng (Allergies)") == 1


class TestUploadAnalysisCacheKey:
    """Regression tests for upload analysis cache key stability."""

    def test_upload_analysis_cache_key_normalizes_type_and_title_spacing(self):
        from app.services.llm import _build_upload_analysis_cache_key

        key_one = _build_upload_analysis_cache_key(
            record_type=" ECG ",
            title="  Kết  quả   Điện tâm đồ ",
            extracted_text="Nhịp xoang đều.",
            image_base64="abc123",
        )
        key_two = _build_upload_analysis_cache_key(
            record_type="ecg",
            title="kết quả điện tâm đồ",
            extracted_text="Nhịp xoang đều.",
            image_base64="abc123",
        )

        assert key_one == key_two

    def test_upload_analysis_cache_key_changes_when_payload_changes(self):
        from app.services.llm import _build_upload_analysis_cache_key

        base = _build_upload_analysis_cache_key(
            record_type="ct",
            title="CT ngực",
            extracted_text="Không thấy tổn thương cấp tính.",
            image_base64=None,
        )
        changed_text = _build_upload_analysis_cache_key(
            record_type="ct",
            title="CT ngực",
            extracted_text="Có nốt mờ nhỏ thùy trên phải.",
            image_base64=None,
        )
        changed_image = _build_upload_analysis_cache_key(
            record_type="ct",
            title="CT ngực",
            extracted_text="Không thấy tổn thương cấp tính.",
            image_base64="different-image",
        )

        assert changed_text != base
        assert changed_image != base


class TestMedicalRecordSchema:
    """Schema validation for structured AI analysis payloads."""

    def test_medical_record_create_validates_structured_analysis(self):
        from app.models.schemas import MedicalRecordCreate

        record = MedicalRecordCreate.model_validate(
            {
                "patient_id": "8d3eb85e-df69-47ac-8473-2c0cb4d30f7f",
                "record_type": "ecg",
                "title": "ECG upload",
                "analysis_result": {
                    "status": "completed",
                    "summary": "ECG nhìn chung ổn định.",
                    "prediction_scores": [
                        {"class": "NORM", "description": "Normal ECG", "score": 0.82},
                    ],
                },
            }
        )

        assert record.analysis_result is not None
        assert record.analysis_result.summary == "ECG nhìn chung ổn định."

    def test_medical_record_create_rejects_invalid_analysis_dict(self):
        from app.models.schemas import MedicalRecordCreate

        with pytest.raises(ValidationError):
            MedicalRecordCreate.model_validate(
                {
                    "patient_id": "8d3eb85e-df69-47ac-8473-2c0cb4d30f7f",
                    "record_type": "ecg",
                    "title": "ECG upload",
                    "analysis_result": {
                        "status": "completed",
                        "summary": "ECG nhìn chung ổn định.",
                        "prediction_scores": [
                            {"class": "NORM", "description": "Normal ECG", "score": 3.5},
                        ],
                    },
                }
            )

    def test_medical_record_create_keeps_legacy_analysis_string(self):
        from app.models.schemas import MedicalRecordCreate

        record = MedicalRecordCreate.model_validate(
            {
                "patient_id": "8d3eb85e-df69-47ac-8473-2c0cb4d30f7f",
                "record_type": "notes",
                "title": "Legacy note",
                "analysis_result": "Legacy plain-text analysis",
            }
        )

        assert record.analysis_result == "Legacy plain-text analysis"


class TestECGUploadAnalysis:
    """Regression tests for the ECG-specific upload analysis branch."""

    def test_extract_json_object_handles_fenced_json(self):
        from app.services.llm import _extract_json_object

        raw = """
Here is the analysis:
```json
{
  "summary": "ECG bình thường",
  "key_findings": ["Không thấy bất thường cấp tính"],
  "urgency": "low"
}
```
"""

        parsed = _extract_json_object(raw)

        assert parsed is not None
        assert parsed["summary"] == "ECG bình thường"
        assert parsed["urgency"] == "low"

    def test_repair_upload_analysis_payload_extracts_sectioned_text(self):
        from app.services.llm import _repair_upload_analysis_payload

        raw = """
Summary: ECG gợi ý thay đổi ST-T nhẹ, chưa thấy dấu hiệu cấp cứu rõ.
Key findings:
- Thay đổi ST-T nhẹ ở các chuyển đạo trước tim.
- Nhịp đều.
Clinical significance: Cần đối chiếu triệu chứng lâm sàng.
Recommended follow-up:
- Theo dõi triệu chứng đau ngực.
- Khám tim mạch nếu còn triệu chứng.
Urgency: medium
Confidence: low
Limitations:
- Đánh giá dựa trên ảnh tải lên.
"""

        repaired = _repair_upload_analysis_payload(None, raw)

        assert repaired["summary"].startswith("ECG gợi ý thay đổi ST-T nhẹ")
        assert repaired["urgency"] == "medium"
        assert repaired["confidence"] == "low"
        assert len(repaired["key_findings"]) == 2
        assert len(repaired["recommended_follow_up"]) == 2

    def test_validated_upload_analysis_result_uses_schema_aliases(self):
        from app.services.llm import _validated_upload_analysis_result

        result = _validated_upload_analysis_result(
            {
                "status": "completed",
                "summary": "ECG nhìn chung ổn định.",
                "key_findings": ["Không thấy biến đổi ST cấp tính."],
                "prediction_scores": [
                    {"class": "NORM", "description": "Normal ECG", "score": 0.82},
                ],
                "model": "test-model",
                "record_type": "ecg",
                "generated_at": "2026-03-16T00:00:00+00:00",
                "request_id": "abc12345",
            },
            fallback_message="Không thể tạo AI analysis.",
        )

        assert result["status"] == "completed"
        assert result["prediction_scores"][0]["class"] == "NORM"
        assert "class_name" not in result["prediction_scores"][0]

    def test_resolve_upload_analysis_summary_rejects_garbled_plain_text(self):
        from app.services.llm import _resolve_upload_analysis_summary

        raw = (
            "25,25,25, and 25, and10,25, and25, and10, and25, and25, and10, "
            "and25, and25, and25, and25, and25, and25, and25, and, and, and, and."
        )

        summary = _resolve_upload_analysis_summary(
            {},
            raw,
            fallback_message="Không thể tạo AI analysis.",
        )

        assert summary == "Không thể tạo AI analysis."

    def test_validated_upload_analysis_result_falls_back_on_schema_error(self):
        from app.services.llm import _validated_upload_analysis_result

        result = _validated_upload_analysis_result(
            {
                "status": "completed",
                "summary": "ECG nhìn chung ổn định.",
                "prediction_scores": [
                    {"class": "NORM", "description": "Normal ECG", "score": 1.4},
                ],
                "model": "test-model",
                "record_type": "ecg",
                "generated_at": "2026-03-16T00:00:00+00:00",
                "request_id": "abc12345",
            },
            fallback_message="Không thể tạo AI analysis.",
        )

        assert result["status"] == "error"
        assert result["summary"] == "Không thể tạo AI analysis."
        assert result["limitations"] == ["LLM returned invalid structured output."]

    @pytest.mark.asyncio
    async def test_ecg_analysis_accepts_adapted_classifier_payload(self, monkeypatch):
        from app.services import llm as llm_module

        async def fake_get_cached_upload_analysis(cache_key):
            return None

        async def fake_store_upload_analysis_cache(cache_key, result):
            return None

        async def fake_check_model_available(model):
            return True

        async def fake_generate(**kwargs):
            return (
                '{"summary":"Tóm tắt ECG","key_findings":["Bất thường ST-T"],'
                '"clinical_significance":"Có ý nghĩa lâm sàng",'
                '"recommended_follow_up":["Khám tim mạch"],'
                '"urgency":"medium","confidence":"high","limitations":["Dựa trên ảnh tải lên"]}'
            )

        async def fake_predict_from_base64(image_base64):
            return {
                "classifier_type": "medsiglip_similarity",
                "checkpoint_path": "remote-score-endpoint",
                "medsiglip_model_id": "google/medsiglip-448",
                "classes": ["NORM", "MI", "STTC", "CD", "HYP"],
                "scores": [0.12, 0.21, 0.83, 0.34, 0.45],
                "scores_by_class": {
                    "NORM": 0.12,
                    "MI": 0.21,
                    "STTC": 0.83,
                    "CD": 0.34,
                    "HYP": 0.45,
                },
                "predicted_labels": ["STTC"],
                "threshold": 0.5,
            }

        monkeypatch.setattr(llm_module, "_get_cached_upload_analysis", fake_get_cached_upload_analysis)
        monkeypatch.setattr(llm_module, "_store_upload_analysis_cache", fake_store_upload_analysis_cache)
        monkeypatch.setattr(llm_module.llm_client, "check_model_available", fake_check_model_available)
        monkeypatch.setattr(llm_module.llm_client, "generate", fake_generate)
        monkeypatch.setattr(llm_module.llm_client, "unload", fake_check_model_available)
        monkeypatch.setattr(llm_module.ecg_classifier_service, "predict_from_base64", fake_predict_from_base64)

        result = await llm_module.analyze_uploaded_record(
            record_type="ecg",
            title="ECG",
            extracted_text=None,
            image_base64="ZmFrZS1pbWFnZQ==",
        )

        assert result["status"] == "completed"
        assert result["summary"] == "Tóm tắt ECG"
        assert result["prediction_scores"][2]["class"] == "STTC"
        assert result["prediction_scores"][2]["score"] == 0.83
        assert result["ecg_classifier"]["classifier_type"] == "medsiglip_similarity"
        assert result["ecg_classifier"]["predicted_labels"] == ["STTC"]

    @pytest.mark.asyncio
    async def test_ecg_analysis_marks_invalid_model_output_as_error(self, monkeypatch):
        from app.services import llm as llm_module

        async def fake_get_cached_upload_analysis(cache_key):
            return None

        async def fake_store_upload_analysis_cache(cache_key, result):
            return None

        async def fake_check_model_available(model):
            return True

        async def fake_generate(**kwargs):
            return "25,25,25, and 25, and10,25, and25, and10, and25, and25, and10"

        async def fake_predict_from_base64(image_base64):
            return {
                "classifier_type": "medsiglip_similarity",
                "checkpoint_path": "remote-score-endpoint",
                "medsiglip_model_id": "google/medsiglip-448",
                "classes": ["NORM", "MI", "STTC", "CD", "HYP"],
                "scores": [-7.508, -7.426, -7.516, -7.637, -7.398],
                "scores_by_class": {
                    "NORM": -7.508,
                    "MI": -7.426,
                    "STTC": -7.516,
                    "CD": -7.637,
                    "HYP": -7.398,
                },
                "predicted_labels": [],
                "threshold": 0.5,
            }

        monkeypatch.setattr(llm_module, "_get_cached_upload_analysis", fake_get_cached_upload_analysis)
        monkeypatch.setattr(llm_module, "_store_upload_analysis_cache", fake_store_upload_analysis_cache)
        monkeypatch.setattr(llm_module.llm_client, "check_model_available", fake_check_model_available)
        monkeypatch.setattr(llm_module.llm_client, "generate", fake_generate)
        monkeypatch.setattr(llm_module.llm_client, "unload", fake_check_model_available)
        monkeypatch.setattr(llm_module.ecg_classifier_service, "predict_from_base64", fake_predict_from_base64)

        result = await llm_module.analyze_uploaded_record(
            record_type="ecg",
            title="ECG",
            extracted_text=None,
            image_base64="ZmFrZS1pbWFnZQ==",
        )

        assert result["status"] == "error"
        assert result["summary"] == "Không thể tạo AI analysis."
        assert result["limitations"] == ["LLM returned invalid structured output."]
        assert len(result["prediction_scores"]) == 5
        assert "ecg_classifier" in result

    @pytest.mark.asyncio
    async def test_text_upload_analysis_repairs_non_json_sectioned_output(self, monkeypatch):
        from app.services import llm as llm_module

        async def fake_get_cached_upload_analysis(cache_key):
            return None

        async def fake_store_upload_analysis_cache(cache_key, result):
            return None

        async def fake_check_model_available(model):
            return True

        async def fake_generate(**kwargs):
            assert "Return plain text using exactly this structure:" in kwargs["prompt"]
            assert "Do not return JSON." in kwargs["prompt"]
            assert "Do not include markdown fences." in kwargs["prompt"]
            return """
Summary: Điện tâm đồ nhìn chung ổn định, chưa thấy dấu hiệu cấp cứu rõ.
Key findings:
- Không thấy biến đổi ST chênh lên rõ.
- Nhịp đều.
Clinical significance: Cần kết hợp triệu chứng và đọc ECG chuẩn.
Recommended follow-up:
- Theo dõi lâm sàng.
Urgency: medium
Confidence: low
Limitations:
- Phân tích dựa trên mô tả tự động.
"""

        monkeypatch.setattr(llm_module, "_get_cached_upload_analysis", fake_get_cached_upload_analysis)
        monkeypatch.setattr(llm_module, "_store_upload_analysis_cache", fake_store_upload_analysis_cache)
        monkeypatch.setattr(llm_module.llm_client, "check_model_available", fake_check_model_available)
        monkeypatch.setattr(llm_module.llm_client, "generate", fake_generate)
        monkeypatch.setattr(llm_module.llm_client, "unload", fake_check_model_available)

        result = await llm_module.analyze_uploaded_record(
            record_type="notes",
            title="Structured note",
            extracted_text="Some OCR text",
            image_base64=None,
        )

        assert result["status"] == "completed"
        assert result["urgency"] == "medium"
        assert result["confidence"] == "low"
        assert result["summary"].startswith("Điện tâm đồ nhìn chung ổn định")
        assert result["key_findings"] == [
            "Không thấy biến đổi ST chênh lên rõ.",
            "Nhịp đều.",
        ]

    @pytest.mark.asyncio
    async def test_ecg_analysis_parses_sectioned_text_and_normalizes_scores(self, monkeypatch):
        from app.services import llm as llm_module

        async def fake_get_cached_upload_analysis(cache_key):
            return None

        async def fake_store_upload_analysis_cache(cache_key, result):
            return None

        async def fake_check_model_available(model):
            return True

        async def fake_generate(**kwargs):
            assert "Return plain text using exactly this structure:" in kwargs["prompt"]
            assert "Do not return JSON." in kwargs["prompt"]
            assert "Do not include markdown fences." in kwargs["prompt"]
            return """
Summary: ECG nhìn chung bình thường.
Key findings:
- Không thấy biến đổi ST-T cấp tính.
Clinical significance: Chưa ghi nhận dấu hiệu nguy cơ cao trên ảnh ECG.
Recommended follow-up:
- Theo dõi lâm sàng nếu còn triệu chứng.
Urgency: low
Confidence: medium
Limitations:
- Đánh giá dựa trên ảnh tải lên.
"""

        async def fake_predict_from_base64(image_base64):
            return {
                "classifier_type": "medsiglip_similarity",
                "checkpoint_path": "remote-score-endpoint",
                "medsiglip_model_id": "google/medsiglip-448",
                "classes": ["NORM", "MI", "STTC", "CD", "HYP"],
                "scores": [-7.508, -7.426, -7.516, -7.637, -7.398],
                "scores_by_class": {
                    "NORM": -7.508,
                    "MI": -7.426,
                    "STTC": -7.516,
                    "CD": -7.637,
                    "HYP": -7.398,
                },
                "predicted_labels": [],
                "threshold": 0.5,
            }

        monkeypatch.setattr(llm_module, "_get_cached_upload_analysis", fake_get_cached_upload_analysis)
        monkeypatch.setattr(llm_module, "_store_upload_analysis_cache", fake_store_upload_analysis_cache)
        monkeypatch.setattr(llm_module.llm_client, "check_model_available", fake_check_model_available)
        monkeypatch.setattr(llm_module.llm_client, "generate", fake_generate)
        monkeypatch.setattr(llm_module.llm_client, "unload", fake_check_model_available)
        monkeypatch.setattr(llm_module.ecg_classifier_service, "predict_from_base64", fake_predict_from_base64)

        result = await llm_module.analyze_uploaded_record(
            record_type="ecg",
            title="ECG",
            extracted_text=None,
            image_base64="ZmFrZS1pbWFnZQ==",
        )

        assert result["summary"] == "ECG nhìn chung bình thường."
        assert result["key_findings"] == ["Không thấy biến đổi ST-T cấp tính."]
        assert all(0.0 <= row["score"] <= 1.0 for row in result["prediction_scores"])
        assert sum(row["score"] for row in result["prediction_scores"]) == pytest.approx(1.0, rel=1e-6)
        assert result["ecg_classifier"]["scores"][0] == -7.508
