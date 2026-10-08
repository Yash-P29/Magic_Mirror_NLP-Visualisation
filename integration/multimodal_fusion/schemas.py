from typing import List, Optional, Tuple, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum

# --- Normalized NLP Schemas ---

class NlpIntent(str, Enum):
    EXPLAINING = "explaining"
    DEFINING = "defining"
    DESCRIBING = "describing"
    GIVING_EXAMPLE = "giving_example"
    DEMONSTRATING = "demonstrating"
    COMPARING = "comparing"
    CONTRASTING = "contrasting"
    ANALOGY = "analogy"
    HYPOTHETICAL = "hypothetical"
    REASONING = "reasoning"
    CAUSE_EFFECT = "cause_effect"
    PROCESS_SEQUENCING = "process_sequencing"
    QUESTIONING = "questioning"
    ANSWERING = "answering"
    INSTRUCTING = "instructing"
    EMPHASIZING = "emphasizing"
    CORRECTING = "correcting"
    CLARIFYING = "clarifying"
    SUMMARIZING = "summarizing"
    RECALLING = "recalling"
    INTRODUCING = "introducing"
    TRANSITIONING = "transitioning"
    CONCLUDING = "concluding"
    SPECULATING = "speculating"
    PREDICTING = "predicting"
    CASUAL = "casual"

class SpeakerActivity(BaseModel):
    intent: NlpIntent
    confidence: float

class Entity(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str
    name: str
    class_name: str = Field(alias="class")
    reference: Optional[str] = None
    status: str

class SemanticRelationship(BaseModel):
    subject: str
    predicate: str
    object: str
    type: str
    confidence: Optional[float] = 1.0

class CoreferenceResolution(BaseModel):
    text_reference: str
    resolved_entity: str

class ExplanationState(BaseModel):
    current_topic: str
    active_concept: str
    recent_entities: List[str]

class NormalizedNlpEvent(BaseModel):
    timestamp: float
    is_partial: bool = False
    text_segment: str
    speaker_activity: List[SpeakerActivity] = []
    entities: List[Entity] = []
    relationships: List[SemanticRelationship] = []
    explicit_information: List[str] = []
    inferred_information: List[str] = []
    coreference_resolutions: List[CoreferenceResolution] = []
    explanation_state: Optional[ExplanationState] = None

# --- Normalized Vision Schemas ---

class Gesture(str, Enum):
    POINTING = "pointing"
    RAISED_HAND = "raised_hand"
    NONE = "none"
    UNKNOWN = "unknown"

class VisualOpportunityType(str, Enum):
    ON_HAND = "on_hand"
    POINTING_TARGET = "pointing_target"
    IN_FRONT_OF_PERSON = "in_front_of_person"
    NEAR_PERSON = "near_person"
    BACKGROUND = "background"
    EMPTY_SPACE = "empty_space"

class AnchorInfo(BaseModel):
    available: bool
    type: Optional[VisualOpportunityType] = None
    bbox: Optional[Tuple[float, float, float, float]] = None # normalized
    center: Optional[Tuple[float, float]] = None

class NormalizedVisionEvent(BaseModel):
    timestamp: float
    frame_id: int
    left_hand_gesture: Gesture = Gesture.UNKNOWN
    right_hand_gesture: Gesture = Gesture.UNKNOWN
    motion: str = "none"
    pointing_at_object_id: Optional[int] = None
    visual_anchor: Optional[AnchorInfo] = None

# --- Visual Planner Schemas ---

class ActionType(str, Enum):
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    MOVE = "MOVE"
    ANIMATE = "ANIMATE"
    REMOVE = "REMOVE"
    NONE = "NONE"

class RepresentationType(str, Enum):
    TYPE_2D = "2d"
    TYPE_3D = "3d"
    GRAPH = "graph"
    DIAGRAM = "diagram"
    PROCESS = "process"
    ANIMATION = "animation"
    TEXT = "text"
    OTHER = "other"

class Representation(BaseModel):
    type: RepresentationType
    description: str

class SemanticState(BaseModel):
    entities: List[str] = []
    actions: List[str] = []
    relationships: List[str] = []
    properties: List[str] = []

class AnchorType(str, Enum):
    HAND = "hand"
    POINTING = "pointing"
    OBJECT = "object"
    REGION = "region"
    WORLD = "world"
    NONE = "none"

class SpatialInfo(BaseModel):
    anchored: bool
    anchor_type: AnchorType
    anchor_id: Optional[str] = None
    position: Optional[List[float]] = None # [x, y, z]
    orientation: Optional[List[float]] = None
    scale: Optional[float] = None

class AnimationInfo(BaseModel):
    type: Optional[str] = None
    state: Optional[str] = None

class ConfidenceScore(BaseModel):
    overall: float
    semantic: float
    reference: float
    spatial: float
    opportunity: float

class VisualPlan(BaseModel):
    event_id: str
    timestamp: float
    action: ActionType
    visual_id: Optional[str] = None
    concept: Optional[str] = None
    representation: Optional[Representation] = None
    semantic_state: Optional[SemanticState] = None
    spatial: Optional[SpatialInfo] = None
    animation: Optional[AnimationInfo] = None
    reason: str
    confidence: Optional[ConfidenceScore] = None
