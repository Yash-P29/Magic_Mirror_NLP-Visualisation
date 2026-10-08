import sys
import queue
import time
import numpy as np
from unittest.mock import MagicMock, patch

from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent # sandbox
_PARENT = _ROOT.parent # nlp
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from asr.whisper_transcriber import WhisperTranscriber

def test_transcribe_vad_enabled():
    """Verify vad_filter=True is passed to transcribe()"""
    with patch('asr.whisper_transcriber.WhisperModel') as mock_whisper_cls:
        mock_model = MagicMock()
        mock_whisper_cls.return_value = mock_model
        
        # Mock transcribe generator
        mock_seg = MagicMock()
        mock_seg.text = "hello"
        mock_model.transcribe.return_value = ([mock_seg], None)
        
        transcriber = WhisperTranscriber(model_size="tiny", device="cpu", compute_type="int8")
        audio_data = np.random.randn(16000 * 2).astype(np.float32)
        
        text, latency = transcriber.transcribe(audio_data, beam_size=5)
        
        assert text == "hello"
        mock_model.transcribe.assert_called_once()
        _, kwargs = mock_model.transcribe.call_args
        assert kwargs.get('vad_filter') is True

def test_short_buffers_accumulated_and_silence_skipped():
    """Verify short buffers accumulate and silent audio is skipped without invoking transcribe."""
    with patch('asr.whisper_transcriber.WhisperModel') as mock_whisper_cls:
        mock_model = MagicMock()
        mock_whisper_cls.return_value = mock_model
        
        transcriber = WhisperTranscriber(model_size="tiny", device="cpu", compute_type="int8")
        transcriber.silence_threshold = 0.05
        transcriber.transcribe_interval = 0.1 
        
        audio_queue = queue.Queue()
        text_queue = queue.Queue()
        
        chunk_size = 1600
        for _ in range(30):
            silence_bytes = np.zeros(chunk_size, dtype=np.int16).tobytes()
            audio_queue.put((silence_bytes, None))
            
        transcriber.start_transcribing(audio_queue, text_queue)
        time.sleep(1.0) 
        transcriber.stop_transcribing()
        
        # Verify model.transcribe was NEVER called
        mock_model.transcribe.assert_not_called()
        
        # Verify text queue is empty
        assert text_queue.empty()

def test_non_silent_audio_reaches_transcription():
    """Verify non-silent audio accumulates > 1.5s and invokes transcribe."""
    with patch('asr.whisper_transcriber.WhisperModel') as mock_whisper_cls:
        mock_model = MagicMock()
        mock_whisper_cls.return_value = mock_model
        
        mock_seg = MagicMock()
        mock_seg.text = "test speech"
        mock_model.transcribe.return_value = ([mock_seg], None)
        
        transcriber = WhisperTranscriber(model_size="tiny", device="cpu", compute_type="int8")
        transcriber.silence_threshold = 0.01
        transcriber.transcribe_interval = 0.1 
        
        audio_queue = queue.Queue()
        text_queue = queue.Queue()
        
        chunk_size = 1600
        for _ in range(30):
            loud_bytes = (np.ones(chunk_size, dtype=np.int16) * 10000).tobytes()
            audio_queue.put((loud_bytes, None))
            
        transcriber.start_transcribing(audio_queue, text_queue)
        time.sleep(1.0)
        transcriber.stop_transcribing()
        
        assert mock_model.transcribe.called
        
        found = False
        while not text_queue.empty():
            msg = text_queue.get_nowait()
            if msg.get("text") == "test speech":
                found = True
                
        assert found
