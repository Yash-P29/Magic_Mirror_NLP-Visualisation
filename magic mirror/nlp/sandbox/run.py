import argparse
import os
import json
import time
import queue

from sandbox.config import (
    RUNTIME_OUTPUTS_DIR,
    SANDBOX_OUTPUTS_DIR,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_MIC_DEVICE_INDEX,
    DEFAULT_WHISPER_MODEL,
    DEFAULT_WHISPER_DEVICE,
    DEFAULT_WHISPER_COMPUTE_TYPE
)
from sandbox.context.state_manager import ContextManager
from sandbox.utils.validator import NLPValidator
from sandbox.audio.microphone import MicrophoneInput
from sandbox.asr.whisper_transcriber import WhisperTranscriber, transcribe_live
from sandbox.extraction.qwen_extractor import QwenExtractor
from sandbox.extraction.fast_processor import FastProcessor
from sandbox.ui.live_terminal import LiveTerminalUI

def save_output(output_json, mode):
    timestamp = int(time.time())
    filename = f"nlp_output_{mode}_{timestamp}.json"
    filepath = os.path.join(RUNTIME_OUTPUTS_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=2)
    print(f"\nSaved NLP output to: {filepath}")

    # Persist to nlp_state_latest.json in both runtime_outputs locations
    for d in [RUNTIME_OUTPUTS_DIR, SANDBOX_OUTPUTS_DIR]:
        try:
            os.makedirs(d, exist_ok=True)
            latest_path = os.path.join(d, "nlp_state_latest.json")
            with open(latest_path, 'w', encoding='utf-8') as f:
                json.dump(output_json, f, indent=2)
            print(f"Saved latest state to: {latest_path}")
        except Exception as e:
            print(f"Warning: could not write to {d}: {e}")

def process_text(text, extractor, context_manager, validator, mode="text"):
    print(f"\n--- Processing Text ---\n'{text}'")
    context_str = context_manager.get_context_string()
    
    print("\nExtracting semantic meaning via local Qwen LLM...")
    output, latency = extractor.extract(text, context_str, is_partial=False)
    print(f"Extraction complete ({latency:.2f}s)")
    if hasattr(extractor, "last_metrics") and extractor.last_metrics:
        m = extractor.last_metrics
        print(f"Token Breakdown: {m.get('eval_count', 0)} tokens generated ({m.get('eval_duration_sec', 0):.2f}s) | Prompt eval: {m.get('prompt_eval_count', 0)} tokens ({m.get('prompt_eval_duration_sec', 0):.2f}s)")
    
    if output:
        is_valid, err = validator.validate_output(output)
        if is_valid:
            print("Schema Validation: SUCCESS")
        else:
            print(f"Schema Validation: FAILED\n{err}")
        
        print("\n--- Extracted JSON ---")
        print(json.dumps(output, indent=2))
        
        save_output(output, mode)
        if hasattr(extractor, "save_state"):
            extractor.save_state(output)
        context_manager.add_interaction(text)
    else:
        print("Failed to generate output.")

def run_mic_session(model_name: str, duration: float = 4.0, device_index: int = DEFAULT_MIC_DEVICE_INDEX):
    """Records a single spoken utterance from the microphone and runs extraction."""
    print(f"\n--- Live Microphone Capture (Device {device_index}, {duration:.1f}s) ---")
    asr = WhisperTranscriber(
        model_size=DEFAULT_WHISPER_MODEL,
        device=DEFAULT_WHISPER_DEVICE,
        compute_type=DEFAULT_WHISPER_COMPUTE_TYPE,
        device_index=device_index
    )
    extractor = QwenExtractor(model_name=model_name)
    context_manager = ContextManager()
    validator = NLPValidator()
    
    print("Please speak into your microphone now...")
    spoken_text = asr.transcribe_live(duration=duration, device_index=device_index)
    if not spoken_text:
        print("No speech detected.")
        return
        
    print(f"\nTranscribed Speech: '{spoken_text}'")
    process_text(spoken_text, extractor, context_manager, validator, mode="mic")

def run_live(model_name: str, device_index: int = DEFAULT_MIC_DEVICE_INDEX):
    print("Starting Magic Mirror Live Streaming Pipeline...")
    
    text_queue = queue.Queue()
    nlp_queue = queue.Queue()
    ui_queue = queue.Queue()
    
    mic = MicrophoneInput(device_index=device_index)
    asr = WhisperTranscriber(
        model_size=DEFAULT_WHISPER_MODEL,
        device=DEFAULT_WHISPER_DEVICE,
        compute_type=DEFAULT_WHISPER_COMPUTE_TYPE,
        device_index=device_index
    )
    fast_proc = FastProcessor()
    deep_proc = QwenExtractor(model_name=model_name)
    ui = LiveTerminalUI()
    
    try:
        # Start Threads
        mic.start_listening()
        asr.start_transcribing(mic.audio_queue, text_queue)
        fast_proc.start_processing(text_queue, nlp_queue, ui_queue)
        deep_proc.start_processing(nlp_queue, ui_queue)
        
        # Main Thread UI Loop
        while True:
            ui.process_queue(ui_queue)
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nShutting down pipeline...")
    finally:
        mic.stop_listening()
        asr.stop_transcribing()
        fast_proc.stop_processing()
        deep_proc.stop_processing()
        print("Shutdown complete.")

def main():
    parser = argparse.ArgumentParser(description="Magic Mirror NLP Sandbox (Local Ollama + faster-whisper)")
    parser.add_argument("--text", type=str, help="Process a single text string (Mode B)")
    parser.add_argument("--mic", action="store_true", help="Record a speech chunk from the microphone and extract semantic meaning")
    parser.add_argument("--live", action="store_true", help="Start real-time continuous streaming pipeline")
    parser.add_argument("--duration", type=float, default=4.0, help="Recording duration in seconds for --mic (default: 4.0)")
    parser.add_argument("--device", type=int, default=DEFAULT_MIC_DEVICE_INDEX, help=f"Audio input device index (default: {DEFAULT_MIC_DEVICE_INDEX})")
    parser.add_argument("--model", type=str, default=DEFAULT_OLLAMA_MODEL, help=f"Model to use for QwenExtractor (default: {DEFAULT_OLLAMA_MODEL})")
    args = parser.parse_args()

    if args.live:
        run_live(model_name=args.model, device_index=args.device)
        return

    if args.mic:
        run_mic_session(model_name=args.model, duration=args.duration, device_index=args.device)
        return

    if args.text:
        print("Initializing NLP Sandbox...")
        context_manager = ContextManager()
        validator = NLPValidator()
        print(f"Using local Qwen Extractor ({args.model}) via Ollama...")
        extractor = QwenExtractor(model_name=args.model)
        process_text(args.text, extractor, context_manager, validator, mode="text")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
