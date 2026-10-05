import json

try:
    import jsonschema
    from jsonschema import validate
except ImportError:
    jsonschema = None
    validate = None

from sandbox.config import SCHEMA_PATH

class NLPValidator:
    def __init__(self):
        try:
            with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
                self.schema = json.load(f)
        except Exception as e:
            print(f"Failed to load schema from {SCHEMA_PATH}: {e}")
            self.schema = None

    def validate_output(self, output_json: dict) -> tuple[bool, str | None]:
        """Validates the output dictionary against the NLP schema.
        Returns (is_valid, error_message).
        """
        if not self.schema:
            return False, "Schema not loaded."
            
        if not isinstance(output_json, dict):
            return False, "Output is not a dictionary."

        if validate is not None:
            try:
                validate(instance=output_json, schema=self.schema)
                return True, None
            except jsonschema.exceptions.ValidationError as err:
                return False, err.message
            except Exception as err:
                return False, str(err)
        else:
            # Fallback structural validation checking required keys when jsonschema is not installed
            required_keys = self.schema.get("required", [
                "timestamp", "is_partial", "text_segment", "speaker_activity", 
                "semantic_graph", "explicit_information", "inferred_information",
                "coreference_resolutions", "explanation_state"
            ])
            missing = [k for k in required_keys if k not in output_json]
            if missing:
                return False, f"Missing required schema keys: {missing}"
            return True, None
