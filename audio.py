import audioop
import queue
import threading

import discord

# Discord receive PCM: 48,000 Hz, 16-bit, stereo.
# Gemini Live input: 16,000 Hz, 16-bit, mono.
def pcm48_stereo_to_16k_mono(data: bytes) -> bytes:
    if not data:
        return b""
    mono = audioop.tomono(data, 2, 0.5, 0.5)
    out, _ = audioop.ratecv(mono, 2, 1, 48000, 16000, None)
    return out


def pcm24_mono_to_48k_stereo(data: bytes) -> bytes:
    if not data:
        return b""
    out, _ = audioop.ratecv(data, 2, 1, 24000, 48000, None)
    return audioop.tostereo(out, 2, 1.0, 1.0)


class GeminiAudioSource(discord.AudioSource):
    """Thread-safe streaming PCM source for Discord's audio player.

    Gemini returns 24 kHz mono PCM. Discord's PCM source format is 48 kHz stereo,
    16-bit little-endian. We resample and expose 20 ms frames (3840 bytes).
    """

    FRAME_BYTES = 3840  # 20 ms * 48000 * 2 channels * 2 bytes

    def __init__(self, playback_queue):
        self.playback_queue = playback_queue
        self.frames: queue.Queue[bytes] = queue.Queue(maxsize=250)
        self._buffer = bytearray()
        self._closed = False
        self._lock = threading.Lock()

    def push_gemini_pcm(self, data: bytes):
        if self._closed:
            return
        try:
            converted = pcm24_mono_to_48k_stereo(data)
            if not converted:
                return
            with self._lock:
                self._buffer.extend(converted)
                while len(self._buffer) >= self.FRAME_BYTES:
                    frame = bytes(self._buffer[:self.FRAME_BYTES])
                    del self._buffer[:self.FRAME_BYTES]
                    try:
                        self.frames.put_nowait(frame)
                    except queue.Full:
                        # Drop old audio to keep latency bounded.
                        try:
                            self.frames.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.frames.put_nowait(frame)
                        except queue.Full:
                            pass
        except Exception:
            # Never allow an audio conversion error to kill Discord's playback thread.
            pass

    def read(self) -> bytes:
        if self._closed:
            return b""
        try:
            # Wait up to 100 ms for audio. A short silence frame keeps Discord's
            # player alive while the next Gemini chunk arrives.
            return self.frames.get(timeout=0.1)
        except queue.Empty:
            return b"\x00" * self.FRAME_BYTES

    def is_opus(self) -> bool:
        return False

    def cleanup(self):
        self._closed = True
        with self._lock:
            self._buffer.clear()
        while not self.frames.empty():
            try:
                self.frames.get_nowait()
            except queue.Empty:
                break
