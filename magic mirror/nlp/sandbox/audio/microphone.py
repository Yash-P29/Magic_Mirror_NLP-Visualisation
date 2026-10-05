import time
import threading
import queue
import sounddevice as sd
import numpy as np
from sandbox.config import DEFAULT_MIC_DEVICE_INDEX

class MicrophoneInput:
    def __init__(self, sample_rate=16000, chunk_size=1024, device_index=DEFAULT_MIC_DEVICE_INDEX):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device_index = device_index
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.stream = None

    def record_chunk(self, duration: float = 4.0) -> np.ndarray:
        """Synchronously records a single chunk of audio from the configured microphone device."""
        num_samples = int(duration * self.sample_rate)
        audio = sd.rec(
            num_samples,
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            device=self.device_index
        )
        sd.wait()
        return audio.flatten()

    def start_listening(self):
        """Starts a background audio stream to continuously capture audio chunks."""
        if self.is_listening:
            return
            
        self.is_listening = True
        
        def callback(indata, frames, time_info, status):
            if status:
                print(f"Microphone status: {status}")
            int16_data = (indata[:, 0] * 32767).astype(np.int16)
            self.audio_queue.put((int16_data.tobytes(), time_info.inputBufferAdcTime))

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            device=self.device_index,
            channels=1,
            dtype='float32',
            blocksize=self.chunk_size,
            callback=callback
        )
        self.stream.start()
        print(f"Microphone listening started (device {self.device_index}, samplerate {self.sample_rate}Hz).")

    def stop_listening(self):
        """Stops the audio stream."""
        self.is_listening = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("Microphone listening stopped.")
