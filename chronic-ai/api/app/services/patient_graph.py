"""
Patient Chat Graph using LangGraph.

Simplified, single-LLM-call flow for patient interactions.
Urgency awareness is baked into the reasoning prompt instead of
a separate triage call, reducing Featherless AI API calls from 3 → 1.

Safety behaviour retained:
- Deterministic self-harm / suicide emergency detection (no API needed)
- Deterministic out-of-scope guard (no API needed)
- Circuit breaker + retry on the single LLM call
"""
import base64
import json
import logging
from pathlib import Path
from typing import AsyncGenerator, Literal, Optional, Tuple
from uuid import UUID
import re
import unicodedata

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from app.services.graph_state import (
    PatientChatState,
    create_stage_message,
    FormattedResponse,
    VerificationResult,
    HITLRequest
)
from app.services.llm_client import llm_client, is_openai_compatible_provider
from app.services.json_utils import strip_markdown_code_fence
from app.services.rag import get_patient_context
from app.services.output_formatter import format_response, get_urgency_indicator
from app.services.doctor_graph import _add_paragraph_breaks
from app.services.resilience import (
    retry_async,
    RetryConfig,
    get_circuit_breaker,
    with_circuit_breaker,
    CircuitBreakerOpen,
    create_idk_response,
    detect_uncertainty_in_response,
    safety_audit,
)
from app.db.database import get_supabase
from app.config import settings

logger = logging.getLogger(__name__)

# Circuit breakers for external services
_llm_breaker = get_circuit_breaker("llm_patient", failure_threshold=3, recovery_timeout=60.0)
_db_breaker = get_circuit_breaker("database_patient", failure_threshold=5, recovery_timeout=30.0)

def _llm_retry_config() -> RetryConfig:
    # Avoid nested retries: OpenAI-compatible providers already retry/fallback in llm_client.
    max_attempts = max(int(settings.llm_retry_max_attempts), 1)
    if is_openai_compatible_provider():
        max_attempts = 1
    return RetryConfig(
        max_attempts=max_attempts,
        base_delay=max(float(settings.llm_retry_base_delay), 0.0),
        max_delay=10.0,
        retryable_exceptions=(RuntimeError, TimeoutError, ConnectionError)
    )

# Safety-first override for self-harm/suicide content.
# If any of these appear in the patient query, we always escalate — NO LLM needed.
SELF_HARM_EMERGENCY_KEYWORDS = [
    "suicide",
    "suicidal",
    "kill myself",
    "end my life",
    "self-harm",
    "hurt myself",
    "tự tử",
    "ý định tự tử",
    "muốn chết",
    "không muốn sống",
    "tự hại",
    "hại bản thân",
]

# Deterministic out-of-scope guardrail for patient chat
PATIENT_NON_MEDICAL_REQUEST_KEYWORDS = [
    "bai tho",
    "lam tho",
    "viet tho",
    "poem",
    "poetry",
    "lyrics",
    "bai hat",
    "viet rap",
    "joke",
    "funny",
    "truyen",
    "ke chuyen",
    "story",
    "essay",
    "viet email",
    "email",
    "viet code",
    "lap trinh",
    "thoi tiet",
    "weather",
    "gia co phieu",
    "stock",
    "bong da",
    "football",
]

PATIENT_MEDICAL_SCOPE_REFUSAL_VI = (
    "Tôi chỉ hỗ trợ các câu hỏi về sức khỏe và y tế.\n\n"
    "Vui lòng mô tả triệu chứng, thuốc đang dùng, hoặc câu hỏi liên quan đến tình trạng sức khỏe của bạn."
)

SELF_HARM_ESCALATION_VI = (
    "⚠️ **Cảnh báo khẩn cấp**: Chúng tôi nhận thấy bạn đang trải qua giai đoạn rất khó khăn.\n\n"
    "Vui lòng liên hệ **đường dây hỗ trợ sức khỏe tâm thần** hoặc đến **cơ sở y tế gần nhất ngay lập tức**.\n\n"
    "Bạn không một mình — hãy tìm kiếm sự giúp đỡ ngay bây giờ."
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _contains_self_harm_emergency(text: str) -> bool:
    """Return True when text contains self-harm/suicide emergency signals."""
    if not text:
        return False
    lower_text = text.lower()
    return any(keyword in lower_text for keyword in SELF_HARM_EMERGENCY_KEYWORDS)


def _normalize_scope_text(text: str) -> str:
    """Normalize text for robust keyword matching."""
    normalized = unicodedata.normalize("NFD", (text or "").lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.replace("đ", "d")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _scope_contains_keyword(normalized_query: str, keyword: str) -> bool:
    """Boundary-safe keyword check."""
    if not keyword:
        return False
    return re.search(rf"\b{re.escape(keyword)}\b", normalized_query) is not None


def _is_out_of_scope_patient_query(query: str) -> Tuple[bool, str]:
    """
    Return whether a patient query is outside medical scope.

    Blocks explicit non-medical intent deterministically (for example: poem requests).
    """
    normalized_query = _normalize_scope_text(query)
    if not normalized_query:
        return True, "empty_query"

    if any(_scope_contains_keyword(normalized_query, kw) for kw in PATIENT_NON_MEDICAL_REQUEST_KEYWORDS):
        return True, "non_medical_intent"

    return False, "in_scope"

def create_initial_patient_state(
    patient_id: str,
    query_vi: str,
    image_path: Optional[str] = None
) -> PatientChatState:
    """Create initial state for patient chat."""
    return PatientChatState(
        # Input
        patient_id=patient_id,
        query_vi=query_vi,
        query_en="",
        image_path=image_path,
        scope_guard_blocked=False,
        scope_guard_reason=None,
        
        # Context
        patient_profile={},
        medical_history="",
        
        # Processing
        verification_result=None,
        reasoning_en="",
        
        # Safety (default low — no triage LLM call)
        urgency_level="low",
        escalation_needed=False,
        escalation_reason=None,
        
        # Output
        response_vi="",
        formatted_response=None,
        
        # Meta
        current_stage="initialized",
        progress=0.0,
        errors=[]
    )


# ============================================================================
# NODE FUNCTIONS
# ============================================================================

async def translate_patient_input_node(state: PatientChatState) -> dict:
    """
    Node: Normalize patient input + apply deterministic guards.

    Checks (in order, all without any API call):
    1. Self-harm / suicide emergency → immediate escalation message
    2. Out-of-scope (non-medical) → scope refusal message
    """
    logger.info(f"[PatientGraph] translate_input: {state['query_vi'][:50]}...")
    query_en = state["query_vi"]

    # Guard 1: self-harm emergency (deterministic — always wins)
    if _contains_self_harm_emergency(query_en):
        logger.info("[PatientGraph] translate_input: self-harm emergency detected, escalating")
        safety_audit.log_decision(
            event_type="self_harm_emergency_guard",
            query=query_en,
            decision="emergency",
            confidence=1.0,
            risk_factors=["self-harm/suicide keyword detected"],
            patient_id=state.get("patient_id"),
            human_review_required=True,
        )
        return {
            "query_en": query_en,
            "scope_guard_blocked": True,
            "scope_guard_reason": "self_harm_emergency",
            "reasoning_en": SELF_HARM_ESCALATION_VI,
            "urgency_level": "emergency",
            "escalation_needed": True,
            "escalation_reason": "Phát hiện dấu hiệu tự gây hại hoặc ý định tự tử",
            "current_stage": "scope_blocked",
            "progress": 0.75,
        }

    # Guard 2: out-of-scope non-medical query (deterministic)
    is_out_of_scope, scope_reason = _is_out_of_scope_patient_query(query_en)
    if is_out_of_scope:
        logger.info(f"[PatientGraph] translate_input: blocked by scope guard ({scope_reason})")
        return {
            "query_en": query_en,
            "scope_guard_blocked": True,
            "scope_guard_reason": scope_reason,
            "reasoning_en": PATIENT_MEDICAL_SCOPE_REFUSAL_VI,
            "current_stage": "scope_blocked",
            "progress": 0.75,
        }

    return {
        "query_en": query_en,
        "scope_guard_blocked": False,
        "scope_guard_reason": None,
        "current_stage": "translated_input",
        "progress": 0.15
    }


async def get_patient_history_node(state: PatientChatState) -> dict:
    """Node: Retrieve patient profile and history (DB only, no LLM)."""
    logger.info(f"[PatientGraph] get_history: Loading profile for {state['patient_id']}...")
    
    supabase = get_supabase()
    
    # Get profile
    patient_result = supabase.table("patients").select("*").eq(
        "id", state["patient_id"]
    ).single().execute()
    
    if not patient_result.data:
        raise ValueError(f"Patient {state['patient_id']} not found")
        
    patient = patient_result.data
    
    # Get relevant medical context via RAG
    context = await get_patient_context(
        patient_id=UUID(state["patient_id"]),
        # Use original Vietnamese query for better retrieval
        query=state["query_vi"],
        max_chunks=3  # Less context for patient to reduce latency
    )
    
    return {
        "patient_profile": patient,
        "medical_history": context,
        "current_stage": "retrieved_history",
        "progress": 0.40
    }


async def patient_reasoning_node(state: PatientChatState) -> dict:
    """
    Node: Generate helpful Vietnamese medical advice.

    Single LLM call — handles both urgency assessment and response generation.
    Replaces the old separate symptom_triage + patient_reasoning calls.
    """
    logger.info("[PatientGraph] reasoning: Generating advice...")

    system_prompt = """You are a supportive medical AI assistant for patients in Vietnam.

URGENCY ASSESSMENT (built-in):
Before responding, silently assess the urgency of the patient's symptoms:
- EMERGENCY: Life-threatening signs (chest pain, difficulty breathing, stroke signs, heavy bleeding, loss of consciousness) → ALWAYS tell them to go to the emergency room immediately as the FIRST thing in your response
- HIGH: Severe symptoms (high fever >39°C, severe pain, significant bleeding) → Strongly advise seeing a doctor within 24 hours as the FIRST thing in your response
- MEDIUM: Bothersome symptoms (persistent symptoms, moderate discomfort) → Recommend making an appointment soon
- LOW: General health questions or mild symptoms → Answer helpfully with standard advice to consult a doctor

FORMATTING (very important):
- Greet the patient warmly using their name at the start
- Use **bold** for key medical terms, medication names, or important values
- Break your answer into short paragraphs separated by blank lines
- Use bullet lists (- item) when listing multiple things (e.g. medications, symptoms, advice)
- DO NOT write one continuous block of text
- Keep language simple, warm, and easy to understand
- Respond ENTIRELY in Vietnamese

CRITICAL GUIDELINES:
- Be empathetic and clear
- Use simple, non-technical language
- ALWAYS remind them to check with their doctor for final decisions
- Do NOT prescribe medication or specific dosages
- Do NOT give definitive diagnoses
- Refuse non-medical requests (poems, stories, entertainment, coding, weather, finance)
- If you are uncertain or don't have enough information, say "Tôi không có đủ thông tin để trả lời chính xác"
- NEVER make up information — only use what's in the patient context

IMPORTANT: If the query is completely outside your medical knowledge or you're very uncertain, respond with:
"Tôi không thể đưa ra lời khuyên cụ thể về vấn đề này. Vui lòng tham khảo ý kiến bác sĩ." """

    prompt = f"""Patient: {state['patient_profile'].get('full_name')}
Age: {state['patient_profile'].get('age')}
Conditions: {state['patient_profile'].get('chronic_conditions', 'Not specified')}
Medical Context: {state['medical_history']}

Query: {state['query_en']}"""

    try:
        async def _reasoning_call():
            return await llm_client.generate(
                model=settings.medical_model,
                prompt=prompt,
                system=system_prompt,
                stream=False
            )

        response_en = await with_circuit_breaker(
            _llm_breaker,
            retry_async,
            _reasoning_call,
            config=_llm_retry_config(),
            operation_name="patient_reasoning"
        )

        # Check for uncertainty in response
        if detect_uncertainty_in_response(response_en, language="vi"):
            logger.info("[PatientGraph] Detected uncertainty in response - adding disclaimer")
            response_en += "\n\n⚠️ Lưu ý: Đây chỉ là thông tin tham khảo chung. Vui lòng trao đổi trực tiếp với bác sĩ để được tư vấn phù hợp."

    except CircuitBreakerOpen:
        logger.error("[PatientGraph] LLM circuit breaker open")
        response_en = create_idk_response(
            reason="Dịch vụ AI y tế đang tạm thời không khả dụng",
            original_query=state['query_en'],
            language="vi",
            suggestions=[
                "Vui lòng thử lại sau vài phút",
                "Nếu khẩn cấp, hãy liên hệ bác sĩ hoặc cơ sở y tế gần nhất",
                "Đến phòng khám hoặc bệnh viện nếu triệu chứng nặng hơn"
            ]
        )

    except Exception as e:
        logger.error(f"[PatientGraph] Reasoning failed: {e}")
        response_en = create_idk_response(
            reason="Đã xảy ra lỗi khi xử lý câu hỏi của bạn",
            original_query=state['query_en'],
            language="vi",
            suggestions=[
                "Vui lòng diễn đạt lại câu hỏi",
                "Mô tả cụ thể hơn về triệu chứng của bạn",
                "Liên hệ bác sĩ nếu triệu chứng kéo dài"
            ]
        )

    # Insert line break after Vietnamese greeting (e.g. "Chào chị Trần Thị Bình,")
    response_en = re.sub(
        r'^(Chào\s+(?:chị|anh|bạn|em|cô|chú|bác)\s+[^,\n]+,)\s*',
        r'\1\n\n',
        response_en,
        count=1,
        flags=re.IGNORECASE
    )

    # Apply paragraph breaks for readability (safety net for dense LLM output)
    response_en = _add_paragraph_breaks(response_en)
    logger.info("[PatientGraph] reasoning: Applied paragraph formatting to response")

    return {
        "reasoning_en": response_en,
        "current_stage": "reasoned",
        "progress": 0.80
    }


async def format_patient_output_node(state: PatientChatState) -> dict:
    """Node: Format output."""
    formatted = format_response(
        response_text=state["reasoning_en"],
        language="vi",
        confidence=0.9 if not state["escalation_needed"] else 1.0
    )
    
    return {
        "formatted_response": formatted,
        "current_stage": "formatted",
        "progress": 0.90
    }


async def translate_patient_output_node(state: PatientChatState) -> dict:
    """Node: Finalize response payload."""
    logger.info("[PatientGraph] translate_output: Finalizing response...")

    response_vi = state["reasoning_en"]

    return {
        "response_vi": response_vi,
        "current_stage": "complete",
        "progress": 1.0
    }


# ============================================================================
# ROUTING
# ============================================================================

def route_after_patient_translation(state: PatientChatState) -> Literal["get_history", "format_output"]:
    """
    Short-circuit to format_output if scope guard or self-harm emergency blocked the query.
    Otherwise proceed to fetch patient history.
    """
    if state.get("scope_guard_blocked"):
        return "format_output"
    return "get_history"


# ============================================================================
# GRAPH BUILDER
# ============================================================================

# Global checkpointer for state persistence
_patient_checkpointer = MemorySaver()


def build_patient_graph():
    """
    Build the simplified patient chat graph.

    Flow:
      START → translate_input → [scope/emergency? → format_output]
                               ↓ (normal)
                            get_history → patient_reasoning → format_output → translate_output → END

    Only ONE external LLM API call (patient_reasoning).
    All guards are deterministic keyword checks.

    Returns:
        Compiled StateGraph ready for execution with state persistence
    """
    builder = StateGraph(PatientChatState)

    # Add nodes
    builder.add_node("translate_input", translate_patient_input_node)
    builder.add_node("get_history", get_patient_history_node)
    builder.add_node("patient_reasoning", patient_reasoning_node)
    builder.add_node("format_output", format_patient_output_node)
    builder.add_node("translate_output", translate_patient_output_node)

    # Define edges
    builder.add_edge(START, "translate_input")
    builder.add_conditional_edges(
        "translate_input",
        route_after_patient_translation,
        ["get_history", "format_output"]
    )
    builder.add_edge("get_history", "patient_reasoning")
    builder.add_edge("patient_reasoning", "format_output")
    builder.add_edge("format_output", "translate_output")
    builder.add_edge("translate_output", END)

    # Compile with checkpointer for state persistence
    return builder.compile(checkpointer=_patient_checkpointer)


# ============================================================================
# PUBLIC API
# ============================================================================

_patient_graph = None


def get_patient_graph():
    """Get or create the patient chat graph singleton."""
    global _patient_graph
    if _patient_graph is None:
        _patient_graph = build_patient_graph()
    return _patient_graph


async def process_patient_chat_graph(
    patient_id: str,
    query_vi: str,
    image_path: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """
    Process patient chat using the LangGraph orchestrator.

    Yields stage updates for real-time UI feedback.

    Args:
        patient_id: ID of the patient
        query_vi: Vietnamese query from patient
        image_path: Optional path to image
        thread_id: Optional thread ID for state persistence

    Yields:
        Stage update dictionaries
    """
    graph = get_patient_graph()
    initial_state = create_initial_patient_state(patient_id, query_vi, image_path)

    # Config for checkpointing - critical for proper state retrieval
    config = {"configurable": {"thread_id": thread_id or f"patient_{patient_id}_{int(__import__('time').time())}"}}

    # Send initial status
    yield create_stage_message("starting", "Đang xử lý câu hỏi của bạn...", 0.05)

    try:
        # Run graph and stream updates
        async for event in graph.astream(initial_state, config=config):
            for node_name, node_output in event.items():
                if node_name == "__interrupt__":
                    yield {
                        "stage": "hitl_required",
                        "hitl_request": node_output[0].value if node_output else None
                    }
                    continue

                if isinstance(node_output, dict):
                    stage = node_output.get("current_stage", "processing")
                    progress = node_output.get("progress", 0.0)

                    stage_messages = {
                        "scope_blocked": "Chỉ hỗ trợ câu hỏi sức khỏe và y tế",
                        "translated_input": "Đã hiểu câu hỏi của bạn",
                        "retrieved_history": "Đã xem xét hồ sơ y tế",
                        "reasoned": "Đang chuẩn bị câu trả lời",
                        "formatted": "Đang định dạng phản hồi",
                        "complete": "Hoàn thành"
                    }

                    yield {
                        "stage": stage,
                        "progress": progress,
                        "message": stage_messages.get(stage, "Đang xử lý...")
                    }

        # CORRECT: Use graph.get_state() to retrieve final accumulated state
        final_state = graph.get_state(config).values

        if final_state and final_state.get("response_vi"):
            urgency_level = final_state.get("urgency_level", "low")
            escalation_needed = final_state.get("escalation_needed", False)

            yield {
                "stage": "complete",
                "message": "Hoàn thành",
                "progress": 1.0,
                "response": final_state["response_vi"],
                "formatted_response": final_state.get("formatted_response"),
                "urgency_level": urgency_level,
                "escalation_needed": escalation_needed,
                "escalation_reason": final_state.get("escalation_reason"),
            }
        else:
            logger.error("[PatientGraph] Final state missing response_vi")
            yield {
                "stage": "error",
                "message": "Xin lỗi, không thể xử lý câu hỏi của bạn. Vui lòng thử lại.",
                "error": "Failed to generate response"
            }

    except CircuitBreakerOpen as e:
        logger.error(f"[PatientGraph] Circuit breaker open: {e}")
        yield {
            "stage": "error",
            "message": "Hệ thống đang tạm thời quá tải. Vui lòng thử lại sau ít phút hoặc liên hệ bác sĩ trực tiếp nếu cần hỗ trợ khẩn cấp.",
            "error": str(e)
        }

    except Exception as e:
        logger.exception(f"[PatientGraph] Error processing query: {e}")

        error_message = "Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn. "
        if "timeout" in str(e).lower():
            error_message += "Hệ thống đang phản hồi chậm. Vui lòng thử lại."
        elif "connection" in str(e).lower():
            error_message += "Không thể kết nối đến dịch vụ. Vui lòng kiểm tra kết nối mạng."
        else:
            error_message += "Vui lòng thử lại hoặc liên hệ hỗ trợ nếu lỗi tiếp tục xảy ra."

        yield {
            "stage": "error",
            "message": error_message,
            "error": str(e)
        }
