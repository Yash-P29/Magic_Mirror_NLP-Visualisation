from typing import Optional, List, Tuple
from collections import deque
import logging

from integration.multimodal_fusion.schemas import NormalizedNlpEvent, NormalizedVisionEvent

logger = logging.getLogger(__name__)

class Synchronizer:
    """Aligns NLP and Vision events using a configurable temporal window."""

    def __init__(self, sync_window_ms: float = 1000.0):
        self.sync_window_ms = sync_window_ms
        self.nlp_buffer: deque = deque()
        self.vision_buffer: deque = deque()

    def add_nlp_event(self, event: NormalizedNlpEvent) -> None:
        self.nlp_buffer.append(event)
        self._prune_buffer(self.nlp_buffer, event.timestamp)

    def add_vision_event(self, event: NormalizedVisionEvent) -> None:
        self.vision_buffer.append(event)
        self._prune_buffer(self.vision_buffer, event.timestamp)

    def _prune_buffer(self, buffer: deque, current_timestamp: float) -> None:
        """Remove events older than the sync window relative to the current timestamp."""
        while buffer and (current_timestamp - buffer[0].timestamp) * 1000.0 > self.sync_window_ms:
            buffer.popleft()

    def get_latest_aligned_events(self) -> Tuple[Optional[NormalizedNlpEvent], Optional[NormalizedVisionEvent]]:
        """
        Returns the most recent valid NLP and Vision events if they are within the window.
        Returns None for modalities that are absent.
        """
        nlp_event = self.nlp_buffer[-1] if self.nlp_buffer else None
        vision_event = self.vision_buffer[-1] if self.vision_buffer else None
        return nlp_event, vision_event
