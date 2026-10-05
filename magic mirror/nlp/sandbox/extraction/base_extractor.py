from abc import ABC, abstractmethod

class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, current_text: str, context_text: str, is_partial: bool) -> tuple[dict, float]:
        """
        Extract semantic meaning from the text.
        
        Args:
            current_text: The current sentence or transcript chunk.
            context_text: The rolling history of the conversation for coreference and state.
            is_partial: Whether this is a partial stream or a complete sentence.
            
        Returns:
            A tuple of (output_dict, latency_in_seconds) where output_dict perfectly 
            matches the nlp_schema.json format.
        """
        pass
