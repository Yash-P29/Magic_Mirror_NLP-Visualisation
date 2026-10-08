import logging
from typing import Optional, Callable

from integration.multimodal_fusion.schemas import (
    NormalizedNlpEvent,
    NormalizedVisionEvent,
    VisualPlan,
    ActionType
)
from integration.multimodal_fusion.synchronizer import Synchronizer
from integration.multimodal_fusion.context_manager import ContextManager
from integration.multimodal_fusion.visual_state import VisualStateManager, ActiveVisual
from integration.multimodal_fusion.opportunity_detector import OpportunityDetector
from integration.multimodal_fusion.visual_planner import VisualPlanner

logger = logging.getLogger(__name__)

class MultimodalFusionEngine:
    """The central engine that fuses NLP and Vision streams to produce VisualPlans."""

    def __init__(self, sync_window_ms: float = 1000.0, on_plan_emitted: Optional[Callable[[VisualPlan], None]] = None):
        self.synchronizer = Synchronizer(sync_window_ms=sync_window_ms)
        self.context_manager = ContextManager()
        self.visual_state = VisualStateManager()
        self.on_plan_emitted = on_plan_emitted
        
        self.last_event_id = None

    def process_nlp_update(self, nlp_event: NormalizedNlpEvent) -> Optional[VisualPlan]:
        """Process an incoming NLP event."""
        self.synchronizer.add_nlp_event(nlp_event)
        self.context_manager.update_from_nlp(nlp_event)
        
        # Only trigger planning on full (non-partial) statements or highly confident partials
        if not nlp_event.is_partial:
            return self._trigger_fusion(trigger_source="NLP")
        return None

    def process_vision_update(self, vision_event: NormalizedVisionEvent) -> Optional[VisualPlan]:
        """Process an incoming Vision event."""
        self.synchronizer.add_vision_event(vision_event)
        self.context_manager.update_from_vision(vision_event)
        
        # We might not want to trigger planning for every vision frame,
        # usually Vision supports NLP. But we can trigger it if there's a strong event.
        # For simplicity, we can let NLP drive the primary updates, 
        # but if we wanted vision to drive, we could add conditions here.
        return None

    def _trigger_fusion(self, trigger_source: str) -> Optional[VisualPlan]:
        """Run the fusion logic to potentially emit a visual plan."""
        nlp_event, vision_event = self.synchronizer.get_latest_aligned_events()
        
        if not nlp_event:
            return None

        # 1. Detect Opportunity
        is_opp, conf, reason = OpportunityDetector.evaluate(nlp_event, self.context_manager.get_context(), self.visual_state)
        
        # 2. Plan
        plan = VisualPlanner.plan(nlp_event, self.context_manager, self.visual_state, conf, reason)
        
        # 3. Apply Plan to internal state (if not NONE)
        if plan.action != ActionType.NONE and plan.visual_id:
            if plan.action == ActionType.CREATE:
                self.visual_state.create_visual(ActiveVisual(
                    visual_id=plan.visual_id,
                    concept=plan.concept or "unknown",
                    representation=plan.representation,
                    spatial=plan.spatial,
                    animation=plan.animation
                ))
            elif plan.action in (ActionType.UPDATE, ActionType.MOVE, ActionType.ANIMATE):
                self.visual_state.update_visual(plan.visual_id, {
                    "representation": plan.representation,
                    "spatial": plan.spatial,
                    "animation": plan.animation
                })
            elif plan.action == ActionType.REMOVE:
                self.visual_state.remove_visual(plan.visual_id)

        # Output Plan
        if self.on_plan_emitted:
            self.on_plan_emitted(plan)
            
        return plan
