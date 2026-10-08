import logging
from typing import Tuple

from integration.multimodal_fusion.schemas import NormalizedNlpEvent
from integration.multimodal_fusion.context_manager import MultimodalContext
from integration.multimodal_fusion.visual_state import VisualStateManager

logger = logging.getLogger(__name__)

class OpportunityDetector:
    """Evaluates whether visualization is useful given the current multimodal context."""
    
    # Intents that strongly suggest visualization
    HIGH_VIZ_INTENTS = {
        "explaining", "demonstrating", "comparing", "hypothetical", 
        "cause_effect", "process_sequencing", "instructing"
    }
    
    # Intents that rarely suggest visualization
    LOW_VIZ_INTENTS = {
        "casual", "transitioning", "concluding", "greeting", "filler"
    }

    @classmethod
    def evaluate(cls, 
                 nlp_event: NormalizedNlpEvent, 
                 context: MultimodalContext, 
                 visual_state: VisualStateManager) -> Tuple[bool, float, str]:
        """
        Returns (is_opportunity, confidence, reason).
        """
        # 1. No entities to visualize
        if not nlp_event.entities:
            # Maybe it's a command to an existing visual (e.g. "remove it")
            has_actionable_coref = False
            for coref in nlp_event.coreference_resolutions:
                if visual_state.get_visual(coref.resolved_entity):
                    has_actionable_coref = True
                    break
                    
            if not has_actionable_coref and not any(r.type in ['action', 'spatial'] for r in nlp_event.relationships):
                return False, 0.0, "No new entities or actionable references"

        # 2. Check intent
        intents = [act.intent.value for act in nlp_event.speaker_activity]
        has_high_intent = any(i in cls.HIGH_VIZ_INTENTS for i in intents)
        has_low_intent = any(i in cls.LOW_VIZ_INTENTS for i in intents)
        
        confidence = 0.5
        reason = "Neutral opportunity"
        
        if has_high_intent:
            confidence += 0.3
            reason = f"High visualization intent detected: {intents}"
        elif has_low_intent:
            confidence -= 0.3
            reason = f"Low visualization intent detected: {intents}"

        # 3. Check for spatial or physical language
        has_spatial_rel = any(rel.type == "spatial" for rel in nlp_event.relationships)
        if has_spatial_rel:
            confidence += 0.2
            reason = "Spatial relationship detected"

        # 4. Check vision context
        if context.last_vision_event and context.last_vision_event.visual_anchor and context.last_vision_event.visual_anchor.available:
            confidence += 0.1
            reason += " (Physical anchor available)"

        confidence = max(0.0, min(1.0, confidence))
        
        is_opportunity = confidence >= 0.5
        
        if not is_opportunity:
            reason = "Confidence too low for visualization"
            
        return is_opportunity, confidence, reason
