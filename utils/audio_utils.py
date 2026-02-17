import librosa
import numpy as np

_MAX_AUDIO_SEC = 30
_SAMPLE_RATE = 16000
_MAX_AUDIO_SAMPLES = int(_MAX_AUDIO_SEC * _SAMPLE_RATE)
_DUMMY_WAVE = np.zeros(_MAX_AUDIO_SAMPLES, dtype=np.float32)
_ORIGINAL_LIBROSA_LOAD = librosa.load


def _to_mono(y):
    if y.ndim == 1:
        return y
    if y.ndim == 2:
        if y.shape[0] <= 8 and y.shape[0] < y.shape[1]:
            return np.mean(y, axis=0)
        return np.mean(y, axis=1)
    return y


def safe_librosa_load(path, sr=_SAMPLE_RATE, *args, **kwargs):
    requested_duration = kwargs.get("duration")
    if requested_duration is None:
        kwargs["duration"] = _MAX_AUDIO_SEC
    else:
        try:
            dur = float(requested_duration)
            kwargs["duration"] = _MAX_AUDIO_SEC if dur <= 0 else min(dur, _MAX_AUDIO_SEC)
        except Exception:
            kwargs["duration"] = _MAX_AUDIO_SEC

    kwargs["sr"] = sr

    try:
        y, sr_ret = _ORIGINAL_LIBROSA_LOAD(path, *args, **kwargs)

        if not isinstance(y, np.ndarray):
            y = np.array(y, dtype=np.float32)
        else:
            y = y.astype(np.float32, copy=False)

        y = _to_mono(y)

        if y.shape[0] > _MAX_AUDIO_SAMPLES:
            y = y[:_MAX_AUDIO_SAMPLES]

        return y, sr_ret

    except (MemoryError, ValueError) as e:
        print(f"[safe_librosa_load] memory/size error while loading {getattr(path, 'name', str(path))}: {e}")
        dummy = _DUMMY_WAVE.copy()
        if dummy.shape[0] == 0:
            dummy = np.zeros(16000, dtype=np.float32)
        return dummy, _SAMPLE_RATE

    except Exception as e:
        print(f"[safe_librosa_load] failed to load {getattr(path, 'name', str(path))}: {e}")
        dummy = _DUMMY_WAVE.copy()
        if dummy.shape[0] == 0:
            dummy = np.zeros(16000, dtype=np.float32)
        return dummy, _SAMPLE_RATE
