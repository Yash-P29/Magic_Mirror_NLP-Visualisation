from typing import List, Optional, Dict
from pydantic import BaseModel

from integration.multimodal_fusion.schemas import (
    NormalizedNlpEvent,
    NormalizedVisionEvent,
    Entity,
    SemanticRelationship
)

class MultimodalContext(BaseModel):
    current_topic: str = ""
    active_concept: str = ""
    recent_entities: List[Entity] = []
    recent_relationships: List[SemanticRelationship] = []
    active_speaker_intents: List[str] = []
    coreference_map: Dict[str, str] = {} # pronoun -> entity_id
    last_vision_event: Optional[NormalizedVisionEvent] = None
    last_nlp_timestamp: float = 0.0

class ContextManager:
    """Maintains a rolling multimodal context window."""
    
    def __init__(self):
        self.context = MultimodalContext()

    def update_from_nlp(self, nlp_event: NormalizedNlpEvent) -> None:
        if nlp_event.explanation_state:
            self.context.current_topic = nlp_event.explanation_state.current_topic
            self.context.active_concept = nlp_event.explanation_state.active_concept
            
        # Update entities (keep last 10)
        for entity in nlp_event.entities:
            # check if exists, update or append
            existing = next((e for e in self.context.recent_entities if e.id == entity.id), None)
            if existing:
                self.context.recent_entities.remove(existing)
            self.context.recent_entities.append(entity)
        
        self.context.recent_entities = self.context.recent_entities[-10:]
        
        self.context.recent_relationships = nlp_event.relationships
        self.context.active_speaker_intents = [act.intent.value for act in nlp_event.speaker_activity]
        
        # Update coreferences
        for coref in nlp_event.coreference_resolutions:
            self.context.coreference_map[coref.text_reference.lower()] = coref.resolved_entity
            
        self.context.last_nlp_timestamp = nlp_event.timestamp

    def update_from_vision(self, vision_event: NormalizedVisionEvent) -> None:
        self.context.last_vision_event = vision_event

    def get_context(self) -> MultimodalContext:
        return self.context

    def resolve_reference(self, reference: str) -> Optional[str]:
        """Attempt to resolve a textual reference (e.g., 'it') to an entity ID."""
        reference_lower = reference.lower()
        if reference_lower in self.context.coreference_map:
            return self.context.coreference_map[reference_lower]
            
        # Fallback to the most recent entity if "it" or similar is used and no explicit coref exists
        if reference_lower in ["it", "this", "that", "the"]:
            if self.context.recent_entities:
                return self.context.recent_entities[-1].id
        return None
