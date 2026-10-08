import logging
from typing import Optional, Callable, Any

from integration.adapters.nlp_adapter import NlpAdapter
from integration.adapters.vision_adapter import VisionAdapter
from integration.multimodal_fusion.fusion_engine import MultimodalFusionEngine
from integration.multimodal_fusion.schemas import VisualPlan

logger = logging.getLogger(__name__)

class MultimodalRuntime:
    """The runtime API interface for the Multimodal Integration Layer."""

    def __init__(self, on_plan_emitted: Optional[Callable[[VisualPlan], None]] = None):
        self.engine = MultimodalFusionEngine(on_plan_emitted=on_plan_emitted)

    def process_nlp_update(self, nlp_json_str: str) -> Optional[VisualPlan]:
        """
        Ingest a raw JSON string from the NLP subsystem.
        Returns a VisualPlan if a decision was made.
        """
        try:
            normalized_event = NlpAdapter.adapt(nlp_json_str)
            return self.engine.process_nlp_update(normalized_event)
        except Exception as e:
            logger.error(f"Error processing NLP update: {e}")
            return None

    def process_vision_update(self, scene_state: Any) -> Optional[VisualPlan]:
        """
        Ingest a SceneState object from the Vision subsystem.
        Returns a VisualPlan if a decision was made.
        """
        try:
            normalized_event = VisionAdapter.adapt(scene_state)
            return self.engine.process_vision_update(normalized_event)
        except Exception as e:
            logger.error(f"Error processing Vision update: {e}")
            return None
