import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def get_gemini_api_key():
    """Retrieves the Gemini API key safely from environment variables (deprecated)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    return api_key

# Global configuration constants
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "nlp_schema.json")
RUNTIME_OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "runtime_outputs")
SANDBOX_OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "runtime_outputs")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("DEFAULT_OLLAMA_MODEL", "qwen2.5:1.5b")

DEFAULT_MIC_DEVICE_INDEX = int(os.environ.get("DEFAULT_MIC_DEVICE_INDEX", 1))
DEFAULT_WHISPER_MODEL = os.environ.get("DEFAULT_WHISPER_MODEL", "base.en")
DEFAULT_WHISPER_DEVICE = os.environ.get("DEFAULT_WHISPER_DEVICE", "cpu")
DEFAULT_WHISPER_COMPUTE_TYPE = os.environ.get("DEFAULT_WHISPER_COMPUTE_TYPE", "int8")
DEFAULT_WHISPER_PROMPT = os.environ.get(
    "DEFAULT_WHISPER_PROMPT",
    "Magic Mirror assistant. Milky Way, galaxy, solar system, planet, orbit, stars, universe, atoms, molecules, science."
)

os.makedirs(RUNTIME_OUTPUTS_DIR, exist_ok=True)
os.makedirs(SANDBOX_OUTPUTS_DIR, exist_ok=True)
