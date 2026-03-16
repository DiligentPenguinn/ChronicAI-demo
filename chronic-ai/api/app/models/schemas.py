"""
Pydantic models for ChronicAI API.

These models define the request/response schemas for the API endpoints,
matching the database schema defined in setup_db.sql.
"""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, ConfigDict, field_validator


# ============================================
# ENUMS
# ============================================

class UserRole(str, Enum):
    patient = "patient"
    doctor = "doctor"
    admin = "admin"


class GenderType(str, Enum):
    male = "male"
    female = "female"
    other = "other"


class BloodType(str, Enum):
    A_POSITIVE = "A+"
    A_NEGATIVE = "A-"
    B_POSITIVE = "B+"
    B_NEGATIVE = "B-"
    AB_POSITIVE = "AB+"
    AB_NEGATIVE = "AB-"
    O_POSITIVE = "O+"
    O_NEGATIVE = "O-"
    UNKNOWN = "unknown"


class SmokingStatus(str, Enum):
    never = "never"
    former = "former"
    current = "current"


class AlcoholConsumption(str, Enum):
    none = "none"
    occasional = "occasional"
    moderate = "moderate"
    heavy = "heavy"


class ExerciseFrequency(str, Enum):
    none = "none"
    light = "light"
    moderate = "moderate"
    regular = "regular"


class InsuranceLevel(str, Enum):
    level_1 = "level_1"
    level_2 = "level_2"
    level_3 = "level_3"
    level_4 = "level_4"


class TriagePriority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    urgent = "urgent"


class ProfileStatus(str, Enum):
    active = "active"
    inactive = "inactive"
    deceased = "deceased"
    suspended = "suspended"


class LicenseStatus(str, Enum):
    active = "active"
    suspended = "suspended"
    revoked = "revoked"
    expired = "expired"


class FacilityType(str, Enum):
    commune_health_station = "commune_health_station"
    district_hospital = "district_hospital"
    provincial_hospital = "provincial_hospital"
    central_hospital = "central_hospital"
    private_clinic = "private_clinic"


class VerificationStatus(str, Enum):
    pending = "pending"
    verified = "verified"
    rejected = "rejected"


class RelationshipType(str, Enum):
    primary_care = "primary_care"
    specialist = "specialist"
    consultant = "consultant"


class AssignmentStatus(str, Enum):
    active = "active"
    inactive = "inactive"
    transferred = "transferred"


class GlucoseTiming(str, Enum):
    fasting = "fasting"
    before_meal = "before_meal"
    after_meal = "after_meal"
    random = "random"


class VitalSource(str, Enum):
    self_reported = "self_reported"
    clinic = "clinic"
    hospital = "hospital"
    device = "device"


class RecordType(str, Enum):
    prescription = "prescription"
    lab = "lab"
    xray = "xray"
    ecg = "ecg"
    ct = "ct"
    mri = "mri"
    notes = "notes"
    referral = "referral"


class ConsultationStatus(str, Enum):
    triage = "triage"
    urgent = "urgent"
    stable = "stable"
    resolved = "resolved"
    cancelled = "cancelled"


class LanguagePref(str, Enum):
    vi = "vi"
    en = "en"


# ============================================
# NESTED MODELS (for JSONB fields)
# ============================================

class ChronicCondition(BaseModel):
    """Structure for chronic_conditions JSONB field."""
    icd10_code: str
    name: str
    diagnosed_date: Optional[date] = None
    stage: Optional[str] = None
    notes: Optional[str] = None


class Medication(BaseModel):
    """Structure for current_medications JSONB field."""
    name: str
    dosage: str
    frequency: str
    timing: Optional[str] = None
    prescriber: Optional[str] = None
    start_date: Optional[date] = None


class SurgicalHistory(BaseModel):
    """Structure for surgical_history JSONB items."""
    procedure: str
    date: Optional[date] = None
    facility: Optional[str] = None
    notes: Optional[str] = None


class SpecialtyCertification(BaseModel):
    """Structure for specialty_certifications JSONB items."""
    specialty: str
    certifying_body: str
    certificate_number: Optional[str] = None
    issue_date: Optional[date] = None
    expiry_date: Optional[date] = None


class ConsultationHours(BaseModel):
    """Structure for daily consultation hours."""
    start: str  # HH:MM format
    end: str
    break_start: Optional[str] = None
    break_end: Optional[str] = None


class NotificationPreferences(BaseModel):
    """Structure for notification_preferences JSONB."""
    sms: bool = True
    app: bool = True
    reminders: Optional[dict] = None


# ============================================
# USER MODELS
# ============================================

class UserBase(BaseModel):
    """Base user model for authentication."""
    phone_number: str = Field(..., max_length=20)
    email: Optional[EmailStr] = None
    role: UserRole


class UserCreate(UserBase):
    """Model for creating a new user."""
    password: Optional[str] = Field(None, min_length=8)


class UserResponse(UserBase):
    """Model for user responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    is_active: bool
    is_verified: bool
    last_login: Optional[datetime] = None
    login_count: int = 0
    created_at: datetime
    updated_at: datetime


# ============================================
# PATIENT MODELS
# ============================================

class PatientBase(BaseModel):
    """Base patient model."""
    full_name: str = Field(..., max_length=255)
    date_of_birth: date
    gender: GenderType
    national_id: Optional[str] = Field(None, max_length=20)
    phone_primary: str = Field(..., max_length=20)
    phone_secondary: Optional[str] = Field(None, max_length=20)
    email: Optional[EmailStr] = None
    profile_photo_url: Optional[str] = Field(None, max_length=500)
    
    # Address
    address_street: Optional[str] = None
    address_ward: str = Field(..., max_length=100)
    address_district: str = Field(..., max_length=100)
    address_province: str = Field(..., max_length=100)
    
    # Emergency Contact
    emergency_contact_name: str = Field(..., max_length=255)
    emergency_contact_phone: str = Field(..., max_length=20)
    emergency_contact_relationship: str = Field(..., max_length=50)
    
    # Medical Demographics
    blood_type: BloodType = BloodType.UNKNOWN
    height_cm: Optional[Decimal] = Field(None, ge=0, le=300)
    weight_kg: Optional[Decimal] = Field(None, ge=0, le=500)
    
    # Chronic Conditions
    chronic_conditions: list[ChronicCondition] = []
    primary_diagnosis: Optional[str] = Field(None, max_length=20)
    diagnosis_date: Optional[date] = None
    disease_stage: Optional[str] = Field(None, max_length=50)
    
    # Medications
    current_medications: list[Medication] = []
    medication_adherence_score: Optional[int] = Field(None, ge=1, le=10)
    
    # Medical History
    allergies: list[str] = []
    surgical_history: list[SurgicalHistory] = []
    family_medical_history: Optional[dict] = None
    medical_history: Optional[dict] = None
    immunization_records: Optional[dict] = None
    treatment_history: Optional[dict] = None
    smoking_status: Optional[SmokingStatus] = None
    alcohol_consumption: Optional[AlcoholConsumption] = None
    exercise_frequency: Optional[ExerciseFrequency] = None
    
    # Insurance
    insurance_provider: Optional[str] = Field(None, max_length=100)
    insurance_number: Optional[str] = Field(None, max_length=50)
    insurance_expiry: Optional[date] = None
    insurance_coverage_level: Optional[InsuranceLevel] = None
    
    # Preferences
    preferred_language: LanguagePref = LanguagePref.vi
    notification_preferences: Optional[NotificationPreferences] = None


class PatientCreate(PatientBase):
    """Model for creating a new patient."""
    user_id: UUID


class PatientUpdate(BaseModel):
    """Model for updating patient info (all fields optional)."""
    full_name: Optional[str] = Field(None, max_length=255)
    phone_primary: Optional[str] = Field(None, max_length=20)
    phone_secondary: Optional[str] = Field(None, max_length=20)
    email: Optional[EmailStr] = None
    profile_photo_url: Optional[str] = Field(None, max_length=500)
    address_street: Optional[str] = None
    address_ward: Optional[str] = Field(None, max_length=100)
    address_district: Optional[str] = Field(None, max_length=100)
    address_province: Optional[str] = Field(None, max_length=100)
    emergency_contact_name: Optional[str] = Field(None, max_length=255)
    emergency_contact_phone: Optional[str] = Field(None, max_length=20)
    emergency_contact_relationship: Optional[str] = Field(None, max_length=50)
    height_cm: Optional[Decimal] = Field(None, ge=0, le=300)
    weight_kg: Optional[Decimal] = Field(None, ge=0, le=500)
    chronic_conditions: Optional[list[ChronicCondition]] = None
    current_medications: Optional[list[Medication]] = None
    medication_adherence_score: Optional[int] = Field(None, ge=1, le=10)
    allergies: Optional[list[str]] = None
    surgical_history: Optional[list[SurgicalHistory]] = None
    family_medical_history: Optional[dict] = None
    medical_history: Optional[dict] = None
    immunization_records: Optional[dict] = None
    treatment_history: Optional[dict] = None
    insurance_provider: Optional[str] = Field(None, max_length=100)
    insurance_number: Optional[str] = Field(None, max_length=50)
    insurance_expiry: Optional[date] = None
    notification_preferences: Optional[NotificationPreferences] = None
    triage_priority: Optional[TriagePriority] = None


class PatientResponse(PatientBase):
    """Model for patient responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    user_id: UUID
    bmi: Optional[Decimal] = None
    assigned_doctor_id: Optional[UUID] = None
    last_checkup_date: Optional[date] = None
    next_appointment_date: Optional[datetime] = None
    triage_priority: TriagePriority = TriagePriority.low
    profile_status: ProfileStatus = ProfileStatus.active
    created_at: datetime
    updated_at: datetime


# ============================================
# DOCTOR MODELS
# ============================================

class DoctorBase(BaseModel):
    """Base doctor model."""
    full_name: str = Field(..., max_length=255)
    date_of_birth: date
    gender: GenderType
    national_id: str = Field(..., max_length=20)
    phone_primary: str = Field(..., max_length=20)
    email: EmailStr
    profile_photo_url: Optional[str] = Field(None, max_length=500)
    
    # Credentials
    medical_license_number: str = Field(..., max_length=50)
    license_issue_date: date
    license_expiry_date: Optional[date] = None
    medical_degree: str = Field(..., max_length=100)
    graduation_year: int = Field(..., ge=1900, le=2100)
    medical_school: str = Field(..., max_length=255)
    
    # Specializations
    primary_specialty: str = Field(..., max_length=100)
    secondary_specialties: list[str] = []
    specialty_certifications: list[SpecialtyCertification] = []
    years_of_experience: int = Field(..., ge=0)
    
    # Work Info
    healthcare_facility_id: Optional[UUID] = None
    healthcare_facility_name: str = Field(..., max_length=255)
    facility_type: FacilityType
    position_title: str = Field(..., max_length=100)
    department: Optional[str] = Field(None, max_length=100)
    work_address: Optional[str] = None
    work_phone: Optional[str] = Field(None, max_length=20)
    
    # Availability
    consultation_hours: Optional[dict[str, ConsultationHours]] = None
    max_daily_consultations: Optional[int] = Field(None, ge=1)
    average_consultation_duration: Optional[int] = Field(None, ge=5)
    accepts_new_patients: bool = True
    teleconsultation_enabled: bool = True
    
    # Chronic Disease Focus
    chronic_disease_focus: list[str] = []
    chronic_management_experience: Optional[dict] = None
    
    # Profile
    preferred_language: LanguagePref = LanguagePref.vi
    bio: Optional[str] = None
    languages_spoken: list[str] = ["vi"]


class DoctorCreate(DoctorBase):
    """Model for creating a new doctor."""
    user_id: UUID


class DoctorUpdate(BaseModel):
    """Model for updating doctor info (all fields optional)."""
    phone_primary: Optional[str] = Field(None, max_length=20)
    email: Optional[EmailStr] = None
    profile_photo_url: Optional[str] = Field(None, max_length=500)
    license_expiry_date: Optional[date] = None
    secondary_specialties: Optional[list[str]] = None
    specialty_certifications: Optional[list[SpecialtyCertification]] = None
    healthcare_facility_id: Optional[UUID] = None
    healthcare_facility_name: Optional[str] = Field(None, max_length=255)
    position_title: Optional[str] = Field(None, max_length=100)
    department: Optional[str] = Field(None, max_length=100)
    work_address: Optional[str] = None
    work_phone: Optional[str] = Field(None, max_length=20)
    consultation_hours: Optional[dict[str, ConsultationHours]] = None
    max_daily_consultations: Optional[int] = Field(None, ge=1)
    accepts_new_patients: Optional[bool] = None
    teleconsultation_enabled: Optional[bool] = None
    chronic_disease_focus: Optional[list[str]] = None
    chronic_management_experience: Optional[dict] = None
    bio: Optional[str] = None
    languages_spoken: Optional[list[str]] = None


class DoctorResponse(DoctorBase):
    """Model for doctor responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    user_id: UUID
    license_status: LicenseStatus = LicenseStatus.active
    total_consultations: int = 0
    active_patient_count: int = 0
    average_rating: Optional[Decimal] = None
    total_ratings: int = 0
    response_time_avg_hours: Optional[Decimal] = None
    verification_status: VerificationStatus = VerificationStatus.pending
    verified_at: Optional[datetime] = None
    profile_status: ProfileStatus = ProfileStatus.active
    created_at: datetime
    updated_at: datetime


# ============================================
# HEALTHCARE FACILITY MODELS
# ============================================

class HealthcareFacilityBase(BaseModel):
    """Base healthcare facility model."""
    name: str = Field(..., max_length=255)
    type: FacilityType
    address: str
    ward: str = Field(..., max_length=100)
    district: str = Field(..., max_length=100)
    province: str = Field(..., max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    email: Optional[EmailStr] = None
    bed_count: Optional[int] = Field(None, ge=0)
    services: list[str] = []
    operating_hours: Optional[dict] = None
    emergency_available: bool = False


class HealthcareFacilityCreate(HealthcareFacilityBase):
    """Model for creating a healthcare facility."""
    pass


class HealthcareFacilityResponse(HealthcareFacilityBase):
    """Model for healthcare facility responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    created_at: datetime
    updated_at: datetime


# ============================================
# VITAL SIGNS MODELS
# ============================================

class VitalSignsBase(BaseModel):
    """Base vital signs model."""
    blood_pressure_systolic: Optional[int] = Field(None, ge=50, le=300)
    blood_pressure_diastolic: Optional[int] = Field(None, ge=30, le=200)
    heart_rate: Optional[int] = Field(None, ge=30, le=250)
    blood_glucose: Optional[Decimal] = Field(None, ge=1.0, le=50.0)
    blood_glucose_timing: Optional[GlucoseTiming] = None
    temperature: Optional[Decimal] = Field(None, ge=30.0, le=45.0)
    oxygen_saturation: Optional[int] = Field(None, ge=50, le=100)
    weight_kg: Optional[Decimal] = Field(None, ge=1.0, le=500.0)
    notes: Optional[str] = None
    source: VitalSource = VitalSource.self_reported


class VitalSignsCreate(VitalSignsBase):
    """Model for creating vital signs record."""
    patient_id: UUID
    recorded_by: Optional[UUID] = None


class VitalSignsResponse(VitalSignsBase):
    """Model for vital signs responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    patient_id: UUID
    recorded_at: datetime
    recorded_by: Optional[UUID] = None
    created_at: datetime


# ============================================
# AI ANALYSIS MODELS
# ============================================

class ECGPredictionScore(BaseModel):
    """Structured ECG classifier score for UI display."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    class_name: str = Field(..., alias="class", min_length=1, max_length=32)
    description: Optional[str] = Field(None, max_length=200)
    score: float = Field(..., ge=0.0, le=1.0)

    @field_validator("class_name", "description", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value):
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class ECGClassifierDetails(BaseModel):
    """Diagnostic metadata from the ECG classifier service."""

    model_config = ConfigDict(extra="forbid")

    classifier_type: Optional[str] = Field(None, max_length=100)
    checkpoint_path: Optional[str] = Field(None, max_length=255)
    medsiglip_model_id: Optional[str] = Field(None, max_length=255)
    classes: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    display_scores: list[float] = Field(default_factory=list)
    scores_by_class: dict[str, float] = Field(default_factory=dict)
    predicted_labels: list[str] = Field(default_factory=list)
    threshold: Optional[float] = Field(None, ge=0.0)

    @field_validator(
        "classifier_type",
        "checkpoint_path",
        "medsiglip_model_id",
        mode="before",
    )
    @classmethod
    def _normalize_text_fields(cls, value):
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("classes", "predicted_labels", mode="before")
    @classmethod
    def _normalize_string_lists(cls, value):
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        return [str(item).strip() for item in value if str(item).strip()]

    @field_validator("display_scores")
    @classmethod
    def _validate_display_scores(cls, value):
        if any(score < 0.0 or score > 1.0 for score in value):
            raise ValueError("display_scores must be between 0 and 1")
        return value


class MedicalRecordAIAnalysis(BaseModel):
    """Validated AI analysis payload stored with a medical record."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    status: Optional[Literal["completed", "skipped", "error"]] = None
    summary: Optional[str] = Field(None, max_length=1200)
    key_findings: list[str] = Field(default_factory=list)
    clinical_significance: Optional[str] = Field(None, max_length=1200)
    recommended_follow_up: list[str] = Field(default_factory=list)
    urgency: Optional[Literal["low", "medium", "high"]] = None
    confidence: Optional[Literal["low", "medium", "high"]] = None
    limitations: list[str] = Field(default_factory=list)
    prediction_scores: list[ECGPredictionScore] = Field(default_factory=list)
    ecg_classifier: Optional[ECGClassifierDetails] = None
    model: Optional[str] = Field(None, max_length=255)
    record_type: Optional[str] = Field(None, max_length=50)
    generated_at: Optional[datetime] = None
    request_id: Optional[str] = Field(None, max_length=64)
    doctor_comment: Optional[str] = Field(None, max_length=2000)

    @field_validator(
        "summary",
        "clinical_significance",
        "model",
        "record_type",
        "request_id",
        "doctor_comment",
        mode="before",
    )
    @classmethod
    def _normalize_scalar_text(cls, value):
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("key_findings", "recommended_follow_up", "limitations", mode="before")
    @classmethod
    def _normalize_text_lists(cls, value):
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        return [str(item).strip() for item in value if str(item).strip()]


# ============================================
# MEDICAL RECORD MODELS
# ============================================

class MedicalRecordBase(BaseModel):
    """Base medical record model."""

    model_config = ConfigDict(from_attributes=True)

    record_type: RecordType
    title: Optional[str] = Field(None, max_length=255)
    content_text: Optional[str] = None
    image_path: Optional[str] = Field(None, max_length=500)
    analysis_result: Optional[MedicalRecordAIAnalysis | str] = None
    doctor_comment: Optional[str] = None

    @field_validator("analysis_result", mode="before")
    @classmethod
    def _validate_analysis_result(cls, value: Any):
        if value is None:
            return None
        if isinstance(value, MedicalRecordAIAnalysis):
            return value
        if isinstance(value, dict):
            return MedicalRecordAIAnalysis.model_validate(value)
        if isinstance(value, str):
            text = value.strip()
            return text or None
        raise TypeError("analysis_result must be a validated AI analysis object or non-empty string")

    @field_validator("doctor_comment", mode="before")
    @classmethod
    def _normalize_doctor_comment(cls, value: Any):
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class MedicalRecordCreate(MedicalRecordBase):
    """Model for creating a medical record."""
    patient_id: UUID
    doctor_id: Optional[UUID] = None


class MedicalRecordResponse(MedicalRecordBase):
    """Model for medical record responses."""

    id: UUID
    patient_id: UUID
    doctor_id: Optional[UUID] = None
    is_verified: bool = False
    verified_by: Optional[UUID] = None
    verified_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


# ============================================
# CONSULTATION MODELS
# ============================================

class ConsultationMessage(BaseModel):
    """Structure for consultation messages."""
    role: str  # 'patient', 'doctor', 'ai'
    content: str
    timestamp: datetime
    attachments: list[str] = []


class ConsultationBase(BaseModel):
    """Base consultation model."""
    chief_complaint: Optional[str] = None
    priority: TriagePriority = TriagePriority.medium


class ConsultationCreate(ConsultationBase):
    """Model for creating a consultation."""
    patient_id: UUID
    doctor_id: Optional[UUID] = None


class ConsultationUpdate(BaseModel):
    """Model for updating consultation."""
    messages: Optional[list[ConsultationMessage]] = None
    summary: Optional[str] = None
    clinical_notes: Optional[dict] = None
    status: Optional[ConsultationStatus] = None
    priority: Optional[TriagePriority] = None
    follow_up_required: Optional[bool] = None
    follow_up_date: Optional[date] = None
    follow_up_notes: Optional[str] = None


class ConsultationResponse(ConsultationBase):
    """Model for consultation responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    patient_id: UUID
    doctor_id: Optional[UUID] = None
    messages: list[ConsultationMessage] = []
    summary: Optional[str] = None
    clinical_notes: Optional[dict] = None
    status: ConsultationStatus = ConsultationStatus.triage
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_minutes: Optional[int] = None
    follow_up_required: bool = False
    follow_up_date: Optional[date] = None
    follow_up_notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# ============================================
# DOCTOR-PATIENT ASSIGNMENT MODELS
# ============================================

class DoctorPatientAssignmentCreate(BaseModel):
    """Model for creating doctor-patient assignment."""
    doctor_id: UUID
    patient_id: UUID
    relationship_type: RelationshipType
    assigned_by: Optional[UUID] = None
    notes: Optional[str] = None


class DoctorPatientAssignmentResponse(BaseModel):
    """Model for assignment responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    doctor_id: UUID
    patient_id: UUID
    relationship_type: RelationshipType
    assigned_date: date
    assigned_by: Optional[UUID] = None
    status: AssignmentStatus = AssignmentStatus.active
    notes: Optional[str] = None
    created_at: datetime
