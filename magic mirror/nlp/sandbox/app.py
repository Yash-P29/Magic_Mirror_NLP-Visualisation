import os

try:
    from flask import Flask, render_template, request, jsonify
except ImportError:
    Flask = None
    render_template = None
    request = None
    jsonify = None

from sandbox.config import DEFAULT_OLLAMA_MODEL
from sandbox.context.state_manager import ContextManager
from sandbox.utils.validator import NLPValidator
from sandbox.extraction.qwen_extractor import QwenExtractor

if Flask is not None:
    app = Flask(__name__)
else:
    app = None

# Initialize local Qwen extractor as sole active engine
try:
    qwen_extractor = QwenExtractor(model_name=DEFAULT_OLLAMA_MODEL)
except Exception as e:
    qwen_extractor = None
    print(f"Qwen not available: {e}")

context_manager = ContextManager()
validator = NLPValidator()

if app is not None:
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/process", methods=["POST"])
    def process_text():
        data = request.json or {}
        text = data.get("text", "")
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
            
        print(f"Processing text: {text}")
        context_str = context_manager.get_context_string()
        
        extractor = qwen_extractor
        
        if not extractor:
            return jsonify({"error": "Qwen extractor is not initialized properly."}), 500
            
        try:
            output, latency = extractor.extract(text, context_str, is_partial=False)
            if output:
                is_valid, err = validator.validate_output(output)
                context_manager.add_interaction(text)
                return jsonify({
                    "success": True,
                    "latency": f"{latency:.2f}s",
                    "valid": is_valid,
                    "validation_error": err if not is_valid else None,
                    "output": output
                })
            else:
                return jsonify({"error": "Failed to generate output"}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    if app is None:
        print("Flask is not installed. Please install flask to run the web interface: pip install flask")
    else:
        print("Starting Magic Mirror Temp Web Interface (Local Qwen2.5:1.5b)...")
        print("Go to http://127.0.0.1:5000 in your browser.")
        app.run(debug=True, port=5000)
