import time
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import queue
from sandbox.config import (
    DEFAULT_WHISPER_MODEL,
    DEFAULT_WHISPER_DEVICE,
    DEFAULT_WHISPER_COMPUTE_TYPE,
    DEFAULT_MIC_DEVICE_INDEX,
    DEFAULT_WHISPER_PROMPT
)

class WhisperTranscriber:
    def __init__(
        self,
        model_size: str = DEFAULT_WHISPER_MODEL,
        device: str = DEFAULT_WHISPER_DEVICE,
        compute_type: str = DEFAULT_WHISPER_COMPUTE_TYPE,
        sample_rate: int = 16000,
        device_index: int = DEFAULT_MIC_DEVICE_INDEX,
        initial_prompt: str = DEFAULT_WHISPER_PROMPT
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.initial_prompt = initial_prompt
        
        print(f"Loading faster-whisper model ({self.model_size} on {self.device}, {self.compute_type})...")
        self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        print("faster-whisper model loaded successfully.")
        
        self.is_transcribing = False
        self.thread = None
        
        # Audio buffer for the rolling streaming window
        self.audio_buffer = np.array([], dtype=np.float32)
        self.max_window_seconds = 30
        self.transcribe_interval = 0.5
        self.last_transcribe_time = 0.0
        
        # Silence detection
        self.silence_threshold = 0.01
        self.silence_duration = 0.0
        self.max_silence = 1.0  # seconds of silence before committing buffer

    def transcribe(self, audio_data: np.ndarray, beam_size: int = 5, initial_prompt: str = None) -> tuple[str, float]:
        """
        Transcribes a 1D float32 numpy audio array (normalized to [-1.0, 1.0] at 16kHz).
        Returns a tuple of (transcribed_text, latency_seconds).
        """
        if audio_data is None or len(audio_data) == 0:
            return "", 0.0
            
        start_time = time.time()
        prompt = self.initial_prompt if initial_prompt is None else initial_prompt
        
        # Pass domain prompt to guide acoustic vocabulary recognition
        segments, _ = self.model.transcribe(
            audio_data,
            language='en',
            beam_size=beam_size,
            initial_prompt=prompt,
            vad_filter=True
        )
        text = " ".join(seg.text for seg in segments).strip()
        latency = time.time() - start_time
        return text, latency

    def transcribe_live(self, duration: float = 4.0, device_index: int = None) -> str:
        """
        Records a short audio chunk (~4 seconds) from sounddevice on the specified
        input device index (defaults to device index 1) and returns the transcribed text.
        """
        target_device = self.device_index if device_index is None else device_index
        num_samples = int(duration * self.sample_rate)
        
        print(f"Recording {duration:.1f}s chunk from device index {target_device}...")
        audio_record = sd.rec(
            num_samples,
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            device=target_device
        )
        sd.wait()
        
        audio_flat = audio_record.flatten()
        text, latency = self.transcribe(audio_flat, beam_size=5)
        print(f"Transcribed ({latency:.2f}s): '{text}'")
        return text

    def start_transcribing(self, audio_queue: queue.Queue, text_queue: queue.Queue):
        """Starts background streaming transcription thread consuming from audio_queue."""
        if self.is_transcribing:
            return
            
        self.is_transcribing = True
        self.thread = threading.Thread(
            target=self._transcribe_loop,
            args=(audio_queue, text_queue),
            daemon=True
        )
        self.thread.start()
        print("faster-whisper streaming transcriber started (background thread).")

    def stop_transcribing(self):
        """Stops background streaming transcription thread."""
        self.is_transcribing = False
        if self.thread:
            self.thread.join()
        print("faster-whisper streaming transcriber stopped.")

    def _rms(self, float_audio: np.ndarray) -> float:
        if len(float_audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(float_audio**2)))

    def _transcribe_loop(self, audio_queue: queue.Queue, text_queue: queue.Queue):
        last_chunk_print = 0.0
        min_window_duration = 1.5
        
        while self.is_transcribing:
            try:
                chunk_data, _ = audio_queue.get(timeout=0.1)
                
                # Convert 16-bit PCM bytes to float32 array
                int_data = np.frombuffer(chunk_data, dtype=np.int16)
                float_data = int_data.astype(np.float32) / 32768.0
                
                self.audio_buffer = np.concatenate((self.audio_buffer, float_data))
                
                # Cap the rolling buffer
                max_samples = self.max_window_seconds * self.sample_rate
                if len(self.audio_buffer) > max_samples:
                    self.audio_buffer = self.audio_buffer[-max_samples:]
                
                # Check for silence at the chunk level
                chunk_rms = self._rms(float_data)
                chunk_duration = len(float_data) / self.sample_rate
                if chunk_rms < self.silence_threshold:
                    self.silence_duration += chunk_duration
                else:
                    self.silence_duration = 0.0

                current_time = time.time()
                accumulated_duration = len(self.audio_buffer) / self.sample_rate
                
                # Transcribe if interval has elapsed and we have at least min_window_duration of audio
                if (current_time - self.last_transcribe_time >= self.transcribe_interval and 
                        accumulated_duration >= min_window_duration):
                    
                    buffer_rms = self._rms(self.audio_buffer)
                    
                    if buffer_rms < self.silence_threshold:
                        if current_time - last_chunk_print >= 1.0:
                            print(f"[WHISPER] buffer={accumulated_duration*1000:.1f} ms rms={buffer_rms:.5f}")
                            print("[WHISPER] silence skipped")
                            last_chunk_print = current_time
                        
                        # Clear silent window so we don't stall or accumulate infinitely
                        self.audio_buffer = np.array([], dtype=np.float32)
                        self.silence_duration = 0.0
                        self.last_transcribe_time = current_time
                        continue
                    
                    if current_time - last_chunk_print >= 1.0:
                        print(f"[WHISPER] buffer={accumulated_duration*1000:.1f} ms rms={buffer_rms:.5f}")
                        print(f"[WHISPER] transcribing {accumulated_duration*1000:.1f} ms")
                        last_chunk_print = current_time
                        
                    text, latency = self.transcribe(self.audio_buffer)
                    
                    is_final = False
                    if self.silence_duration > self.max_silence and len(text) > 0:
                        is_final = True
                    
                    if text:
                        text_queue.put({
                            "text": text,
                            "is_partial": not is_final,
                            "latency": latency,
                            "timestamp": current_time
                        })
                    
                    if is_final:
                        self.audio_buffer = np.array([], dtype=np.float32)
                        self.silence_duration = 0.0
                    
                    self.last_transcribe_time = current_time
                
            except queue.Empty:
                continue
            except Exception as e:
                import traceback
                print(f"[WHISPER] exception in loop:\n{traceback.format_exc()}")

def transcribe_live(
    duration: float = 4.0,
    device_index: int = DEFAULT_MIC_DEVICE_INDEX,
    model_size: str = DEFAULT_WHISPER_MODEL
) -> str:
    """Standalone helper function to capture audio from the mic and transcribe using faster-whisper."""
    transcriber = WhisperTranscriber(model_size=model_size, device_index=device_index)
    return transcriber.transcribe_live(duration=duration, device_index=device_index)
