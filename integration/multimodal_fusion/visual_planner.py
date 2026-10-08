import uuid
import logging
from typing import Optional

from integration.multimodal_fusion.schemas import (
    NormalizedNlpEvent,
    VisualPlan,
    ActionType,
    Representation,
    RepresentationType,
    SemanticState,
    SpatialInfo,
    AnchorType,
    AnimationInfo,
    ConfidenceScore,
    VisualOpportunityType
)
from integration.multimodal_fusion.context_manager import ContextManager
from integration.multimodal_fusion.visual_state import VisualStateManager, ActiveVisual

logger = logging.getLogger(__name__)

class VisualPlanner:
    """Generates a VisualPlan based on multimodal context and visual opportunity."""

    @classmethod
    def plan(cls, 
             nlp_event: NormalizedNlpEvent, 
             context_manager: ContextManager, 
             visual_state: VisualStateManager,
             opportunity_confidence: float,
             opportunity_reason: str) -> VisualPlan:
        
        event_id = str(uuid.uuid4())
        context = context_manager.get_context()
        
        # Default plan is NONE
        plan = VisualPlan(
            event_id=event_id,
            timestamp=nlp_event.timestamp,
            action=ActionType.NONE,
            reason=opportunity_reason,
            confidence=ConfidenceScore(
                overall=0.0, semantic=0.0, reference=0.0, spatial=0.0, opportunity=opportunity_confidence
            )
        )

        if opportunity_confidence < 0.5:
            return plan

        # 1. Identify primary entity or referenced entity
        primary_entity_id = None
        target_visual = None
        action_type = ActionType.CREATE
        reference_visual = None
        
        # Check explicit coreferences first (e.g., "it", "this") to establish context
        for coref in nlp_event.coreference_resolutions:
            resolved = context_manager.resolve_reference(coref.text_reference)
            if resolved:
                reference_visual = visual_state.get_visual(resolved)
                if reference_visual:
                    target_visual = reference_visual
                    primary_entity_id = resolved
                    action_type = ActionType.UPDATE
                    break

        # If there are new explicit entities, they take priority as the primary target for CREATE/MOVE
        new_entities = [e for e in nlp_event.entities if e.status == "new_explicit"]
        if new_entities:
            ent = new_entities[0]
            primary_entity_id = ent.id
            if visual_state.get_visual(primary_entity_id) or visual_state.get_visual_by_concept(ent.name):
                action_type = ActionType.UPDATE
                target_visual = visual_state.get_visual(primary_entity_id) or visual_state.get_visual_by_concept(ent.name)
                primary_entity_id = target_visual.visual_id
            else:
                action_type = ActionType.CREATE
                target_visual = None # We are creating it, so it doesn't exist yet
        elif not primary_entity_id and nlp_event.entities:
            # If they are existing/inferred, maybe update
            ent = nlp_event.entities[0]
            primary_entity_id = ent.id
            target_visual = visual_state.get_visual(primary_entity_id) or visual_state.get_visual_by_concept(ent.name)
            if target_visual:
                action_type = ActionType.UPDATE
                primary_entity_id = target_visual.visual_id
            else:
                action_type = ActionType.CREATE

        # Handle specific intent actions (like remove or animate)
        text_lower = nlp_event.text_segment.lower()
        if "remove" in text_lower or "delete" in text_lower or "hide" in text_lower:
            if target_visual:
                action_type = ActionType.REMOVE
        elif "animate" in text_lower or "make it rotate" in text_lower or "make it run" in text_lower or "stand" in text_lower:
             if target_visual:
                 action_type = ActionType.ANIMATE
        elif "move" in text_lower or "put" in text_lower:
             if target_visual:
                 action_type = ActionType.MOVE

        if not primary_entity_id and not target_visual:
            plan.reason = "Could not identify primary entity or reference."
            return plan

        # Extract semantic state
        concept = target_visual.concept if target_visual else next((e.name for e in nlp_event.entities if e.id == primary_entity_id), "unknown")
        
        semantic_state = SemanticState(
            entities=[e.id for e in nlp_event.entities],
            actions=[act.intent.value for act in nlp_event.speaker_activity],
            relationships=[f"{r.subject}_{r.predicate}_{r.object}" for r in nlp_event.relationships]
        )

        # Grounding with Vision
        spatial_info = SpatialInfo(anchored=False, anchor_type=AnchorType.NONE)
        vision = context.last_vision_event
        
        if vision and vision.visual_anchor and vision.visual_anchor.available:
            anchor = vision.visual_anchor
            spatial_info.anchored = True
            
            # Map vision opportunity type to planner anchor type
            if anchor.type == VisualOpportunityType.ON_HAND:
                spatial_info.anchor_type = AnchorType.HAND
            elif anchor.type == VisualOpportunityType.POINTING_TARGET:
                spatial_info.anchor_type = AnchorType.POINTING
            else:
                spatial_info.anchor_type = AnchorType.WORLD

            if anchor.center:
                spatial_info.position = [anchor.center[0], anchor.center[1], 0.0]

        # Determine representation type based on context
        rep_type = RepresentationType.TYPE_3D
        if "molecule" in text_lower or "atom" in text_lower or "scientific" in text_lower:
            rep_type = RepresentationType.TYPE_3D
        elif "graph" in text_lower:
            rep_type = RepresentationType.GRAPH

        representation = Representation(
            type=rep_type,
            description=f"Representation for {concept}"
        )

        animation_info = None
        if action_type == ActionType.ANIMATE:
             # simple extraction
             anim_state = "moving"
             if "rotate" in text_lower: anim_state = "rotating"
             elif "run" in text_lower: anim_state = "running"
             elif "stand" in text_lower: anim_state = "standing"
             
             animation_info = AnimationInfo(type="procedural", state=anim_state)

        # Confidence calculation
        semantic_conf = sum(a.confidence for a in nlp_event.speaker_activity) / max(1, len(nlp_event.speaker_activity))
        ref_conf = 1.0 if target_visual else 0.8
        spatial_conf = 0.9 if spatial_info.anchored else 0.5
        overall_conf = (semantic_conf + ref_conf + spatial_conf + opportunity_confidence) / 4.0

        plan.action = action_type
        plan.visual_id = primary_entity_id
        plan.concept = concept
        plan.representation = representation
        plan.semantic_state = semantic_state
        plan.spatial = spatial_info
        plan.animation = animation_info
        plan.reason = "Visual plan generated successfully based on opportunity."
        plan.confidence = ConfidenceScore(
            overall=overall_conf,
            semantic=semantic_conf,
            reference=ref_conf,
            spatial=spatial_conf,
            opportunity=opportunity_confidence
        )

        return plan
