"""Audio processing engine — captures and analyzes audio for suspicious sounds."""

import time
import threading
import queue
import numpy as np
from typing import Optional, Dict


class AudioProcessor:
    """Captures and processes audio from the microphone."""

    def __init__(self, sample_rate: int = 16000, threshold_db: float = -40.0):
        self.sample_rate = sample_rate
        self.threshold_db = threshold_db
        self._running = False
        self._audio_queue: queue.Queue = queue.Queue(maxsize=10)
        self._audio_thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """Start audio capture."""
        try:
            import sounddevice as sd

            def audio_callback(indata, frames, time_info, status):
                if status:
                    pass  # Handle status if needed
                try:
                    self._audio_queue.put_nowait(indata.copy())
                except queue.Full:
                    pass

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=1024,
                callback=audio_callback,
            )
            self._stream.start()
            self._running = True
            return True
        except ImportError:
            # Fallback: mock audio processing
            self._running = True
            return True

    def get_latest_audio(self, duration_seconds: float = 1.0) -> Optional[np.ndarray]:
        """Get the latest audio chunk."""
        try:
            chunk_size = int(self.sample_rate * duration_seconds)
            chunks = []
            total_samples = 0
            while total_samples < chunk_size and self._running:
                try:
                    chunk = self._audio_queue.get_nowait()
                    chunks.append(chunk)
                    total_samples += len(chunk)
                except queue.Empty:
                    break
            if chunks:
                return np.concatenate(chunks)[-chunk_size:]
            return None
        except Exception:
            return None

    def compute_features(self, audio: np.ndarray) -> Dict:
        """Compute audio features from raw audio data.

        Returns dict with:
            - rms_energy: root mean square energy
            - peak_amplitude: max amplitude
            - volume_db: average volume in dB
            - zero_crossing_rate: ZCR
            - spectral_centroid: spectral centroid (Hz)
        """
        if audio is None or len(audio) == 0:
            return {
                "rms_energy": 0.0, "peak_amplitude": 0.0,
                "volume_db": -100.0, "zero_crossing_rate": 0.0,
                "spectral_centroid": 0.0,
            }

        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))
        db = 20 * np.log10(rms + 1e-10)

        # Zero crossing rate
        zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / 2

        # Spectral centroid (approximate)
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / self.sample_rate)
        centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)

        return {
            "rms_energy": float(rms),
            "peak_amplitude": float(peak),
            "volume_db": float(db),
            "zero_crossing_rate": float(zcr),
            "spectral_centroid": float(centroid),
        }

    def is_suspicious(self, features: Dict) -> bool:
        """Determine if audio features indicate suspicious activity."""
        return features.get("volume_db", -100) > self.threshold_db

    def stop(self) -> None:
        """Stop audio capture."""
        self._running = False
        if hasattr(self, '_stream'):
            self._stream.stop()
            self._stream.close()
