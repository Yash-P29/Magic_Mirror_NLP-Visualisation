from typing import Dict, Optional, List
from pydantic import BaseModel
from integration.multimodal_fusion.schemas import Representation, SpatialInfo, AnimationInfo

class ActiveVisual(BaseModel):
    visual_id: str
    concept: str
    representation: Optional[Representation] = None
    spatial: Optional[SpatialInfo] = None
    animation: Optional[AnimationInfo] = None
    status: str = "active"

class VisualStateManager:
    """Manages the currently active visualizations in the scene."""
    
    def __init__(self):
        self.active_visuals: Dict[str, ActiveVisual] = {}

    def create_visual(self, visual: ActiveVisual) -> None:
        self.active_visuals[visual.visual_id] = visual

    def update_visual(self, visual_id: str, updates: dict) -> None:
        if visual_id in self.active_visuals:
            visual = self.active_visuals[visual_id]
            for key, value in updates.items():
                if hasattr(visual, key) and value is not None:
                    setattr(visual, key, value)

    def remove_visual(self, visual_id: str) -> None:
        if visual_id in self.active_visuals:
            del self.active_visuals[visual_id]

    def get_visual(self, visual_id: str) -> Optional[ActiveVisual]:
        return self.active_visuals.get(visual_id)

    def get_all_active(self) -> List[ActiveVisual]:
        return list(self.active_visuals.values())

    def get_visual_by_concept(self, concept: str) -> Optional[ActiveVisual]:
        # Basic matching, could be improved with semantic similarity
        for vis in self.active_visuals.values():
            if vis.concept.lower() == concept.lower():
                return vis
        return None
