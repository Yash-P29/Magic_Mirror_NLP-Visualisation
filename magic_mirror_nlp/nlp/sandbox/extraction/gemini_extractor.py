import time
import json
from google import genai
from google.genai import types
from sandbox.extraction.base_extractor import BaseExtractor
from sandbox.config import get_gemini_api_key, SCHEMA_PATH

class GeminiExtractor(BaseExtractor):
    def __init__(self, model_name="gemini-2.5-pro"):
        api_key = get_gemini_api_key()
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        
        with open(SCHEMA_PATH, 'r') as f:
            self.schema = json.load(f)
            
        # Clean up schema for Gemini's structured output
        # Remove top level $schema and title which are not supported in response_schema
        if "$schema" in self.schema:
            del self.schema["$schema"]
        if "title" in self.schema:
            del self.schema["title"]
        if "description" in self.schema:
            del self.schema["description"]

        self.system_instruction = """
You are the NLP reasoning engine for Magic Mirror, a live Augmented Reality visualization system.
Your job is to read the CURRENT TEXT spoken by the user and the PREVIOUS CONTEXT.
You must output a structured JSON object representing the semantic graph and discourse intent.

CRITICAL INSTRUCTIONS:
1. Distinguish between explicit statements (actually spoken) and inferred knowledge.
2. Resolve pronouns (it, this, that) using the provided context.
3. Keep track of the active entities and the current topic in the explanation_state.
4. Extract relationships as simple triples: subject -> predicate -> object.
5. Accurately classify the speaker_activity. If it's a casual remark ("Can you get coffee?"), label it as 'casual'.
6. Do NOT invent information.
"""

    def extract(self, current_text: str, context_text: str, is_partial: bool) -> tuple[dict, float]:
        start_time = time.time()
        
        prompt = f"""
{context_text}

CURRENT TEXT TO PROCESS (is_partial={is_partial}):
"{current_text}"
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    response_mime_type="application/json",
                    response_schema=self.schema,
                    temperature=0.0 # Lowest temperature for consistent extraction
                )
            )
            latency = time.time() - start_time
            result_dict = json.loads(response.text)
            
            # Ensure timestamp and is_partial are set accurately regardless of LLM hallucination
            result_dict["timestamp"] = int(time.time())
            result_dict["is_partial"] = is_partial
            result_dict["text_segment"] = current_text
            
            return result_dict, latency
        except Exception as e:
            print(f"Gemini Extraction Error: {e}")
            return {}, 0.0
