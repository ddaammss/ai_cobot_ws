# wakeup.py
import os
import numpy as np
import openwakeword
from openwakeword.model import Model
from scipy.signal import resample
from pathlib import Path

# 상대 경로
BASE_DIR = Path.home() / 'dynamic_busan' / 'src' / 'bartender' / 'bartender' /'stt'
MODEL_PATH = os.path.join(BASE_DIR, "hello_rokey_8332_32.tflite")


class WakeupWord:
    def __init__(self, buffer_size):
        openwakeword.utils.download_models()
        self.model = None
        self.model_name = "hello_rokey_8332_32"
        self.stream = None
        self.buffer_size = buffer_size

    def set_stream(self, stream):
        self.model = Model(wakeword_models=[MODEL_PATH])
        self.stream = stream

    def is_wakeup(self):
        audio_chunk = np.frombuffer(
            self.stream.read(self.buffer_size, exception_on_overflow=False),
            dtype=np.int16,
        )
        audio_chunk = resample(
            audio_chunk,
            int(len(audio_chunk) * 16000 / 48000)
        )
        outputs = self.model.predict(audio_chunk, threshold=0.1)
        confidence = outputs[self.model_name]
        return confidence > 0.3
