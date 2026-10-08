import logging
from typing import Any

from integration.multimodal_fusion.schemas import (
    NormalizedVisionEvent,
    Gesture,
    AnchorInfo,
    VisualOpportunityType
)

logger = logging.getLogger(__name__)

class VisionAdapter:
    """Adapts raw Vision SceneState into NormalizedVisionEvent."""
    
    @staticmethod
    def adapt(scene_state: Any) -> NormalizedVisionEvent:
        # We accept 'Any' to avoid tight coupling to the exact class type if it's imported differently
        try:
            left_gesture = Gesture.UNKNOWN
            right_gesture = Gesture.UNKNOWN
            motion = "none"
            
            if scene_state.activity:
                try:
                    left_gesture = Gesture(scene_state.activity.left_hand_gesture.value)
                except ValueError:
                    left_gesture = Gesture.UNKNOWN
                    
                try:
                    right_gesture = Gesture(scene_state.activity.right_hand_gesture.value)
                except ValueError:
                    right_gesture = Gesture.UNKNOWN
                    
                motion = scene_state.activity.motion

            pointing_at_id = None
            # Extract pointing relationship if it exists
            for rel in scene_state.relationships:
                if type(rel).__name__ == "HandPointingAtObject":
                    pointing_at_id = rel.object_id
                    break

            anchor_info = None
            if scene_state.visual_opportunity:
                vo = scene_state.visual_opportunity
                vo_type = None
                try:
                    vo_type = VisualOpportunityType(vo.type.value)
                except ValueError:
                    pass

                anchor_info = AnchorInfo(
                    available=vo.available,
                    type=vo_type,
                    bbox=vo.bbox,
                    center=vo.center
                )

            return NormalizedVisionEvent(
                timestamp=scene_state.timestamp,
                frame_id=scene_state.frame_id,
                left_hand_gesture=left_gesture,
                right_hand_gesture=right_gesture,
                motion=motion,
                pointing_at_object_id=pointing_at_id,
                visual_anchor=anchor_info
            )
        except Exception as e:
            logger.error(f"Failed to adapt Vision event: {e}")
            raise
