import json
import logging
from typing import Dict, Any

from integration.multimodal_fusion.schemas import (
    NormalizedNlpEvent,
    SpeakerActivity,
    Entity,
    SemanticRelationship,
    CoreferenceResolution,
    ExplanationState,
    NlpIntent
)

logger = logging.getLogger(__name__)

class NlpAdapter:
    """Adapts raw NLP subsystem JSON output into NormalizedNlpEvent."""
    
    @staticmethod
    def adapt(raw_json_str: str) -> NormalizedNlpEvent:
        try:
            data = json.loads(raw_json_str)
            
            speaker_activities = []
            for act in data.get("speaker_activity", []):
                try:
                    speaker_activities.append(SpeakerActivity(
                        intent=NlpIntent(act["intent"]),
                        confidence=act["confidence"]
                    ))
                except ValueError:
                    # Fallback if unknown intent
                    pass

            entities = []
            semantic_graph = data.get("semantic_graph", {})
            for ent in semantic_graph.get("entities", []):
                entities.append(Entity(
                    id=ent.get("id", ""),
                    name=ent.get("name", ""),
                    class_name=ent.get("class", ""),
                    reference=ent.get("reference"),
                    status=ent.get("status", "")
                ))
            
            relationships = []
            for rel in semantic_graph.get("relationships", []):
                relationships.append(SemanticRelationship(
                    subject=rel.get("subject", ""),
                    predicate=rel.get("predicate", ""),
                    object=rel.get("object", ""),
                    type=rel.get("type", ""),
                    confidence=rel.get("confidence", 1.0)
                ))

            coref_resolutions = []
            for coref in data.get("coreference_resolutions", []):
                coref_resolutions.append(CoreferenceResolution(
                    text_reference=coref.get("text_reference", ""),
                    resolved_entity=coref.get("resolved_entity", "")
                ))

            expl_state_raw = data.get("explanation_state", {})
            explanation_state = None
            if expl_state_raw:
                explanation_state = ExplanationState(
                    current_topic=expl_state_raw.get("current_topic", ""),
                    active_concept=expl_state_raw.get("active_concept", ""),
                    recent_entities=expl_state_raw.get("recent_entities", [])
                )

            return NormalizedNlpEvent(
                timestamp=data.get("timestamp", 0.0),
                is_partial=data.get("is_partial", False),
                text_segment=data.get("text_segment", ""),
                speaker_activity=speaker_activities,
                entities=entities,
                relationships=relationships,
                explicit_information=data.get("explicit_information", []),
                inferred_information=data.get("inferred_information", []),
                coreference_resolutions=coref_resolutions,
                explanation_state=explanation_state
            )

        except Exception as e:
            logger.error(f"Failed to adapt NLP event: {e}")
            raise
