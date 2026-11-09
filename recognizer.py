
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fft import dct
from scipy.signal import resample_poly, stft


from paths import BASE_DIR

SOUNDS_PATH = BASE_DIR / "database" / "Sounds"

TARGET_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 220
N_MELS = 64
N_MFCC = 20

# Оценка совпадения: exp(-distance / DISTANCE_SCALE)
DISTANCE_SCALE = 0.25


def load_audio(path):
    audio, sample_rate = sf.read(path, dtype="float32")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    return audio, int(sample_rate)


def hz_to_mel(frequency):
    return 2595.0 * np.log10(1.0 + frequency / 700.0)


def mel_to_hz(value):
    return 700.0 * (10.0 ** (value / 2595.0) - 1.0)


def create_mel_filterbank():
    frequencies = np.linspace(
        0,
        TARGET_RATE / 2,
        N_FFT // 2 + 1
    )

    mel_points = mel_to_hz(
        np.linspace(
            hz_to_mel(0),
            hz_to_mel(TARGET_RATE / 2),
            N_MELS + 2
        )
    )

    filters = np.zeros(
        (N_MELS, len(frequencies)),
        dtype=np.float32
    )

    for index in range(N_MELS):
        left, center, right = mel_points[index:index + 3]

        filters[index] = np.maximum(
            0.0,
            np.minimum(
                (frequencies - left) / (center - left),
                (right - frequencies) / (right - center)
            )
        )

    return filters


MEL_FILTERBANK = create_mel_filterbank()


def extract_features(audio, sample_rate):
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if audio.size == 0:
        return np.empty((0, N_MFCC), dtype=np.float32)

    audio = audio - np.mean(audio)

    if int(sample_rate) != TARGET_RATE:
        divisor = math.gcd(int(sample_rate), TARGET_RATE)

        audio = resample_poly(
            audio,
            TARGET_RATE // divisor,
            int(sample_rate) // divisor
        ).astype(np.float32)

    if len(audio) < 256:
        return np.empty((0, N_MFCC), dtype=np.float32)

    _, _, spectrum = stft(
        audio,
        fs=TARGET_RATE,
        nperseg=N_FFT,
        noverlap=N_FFT - HOP_LENGTH,
        boundary="zeros"
    )

    power = np.abs(spectrum).astype(np.float32) ** 2
    mel_power = MEL_FILTERBANK @ power
    log_mel = np.log(mel_power + 1e-10)

    frame_energy = np.sum(power, axis=0)
    peak_energy = float(frame_energy.max())

    if peak_energy < 1e-12:
        return np.empty((0, N_MFCC), dtype=np.float32)

    # Убираем длинную тишину по краям.
    active_frames = frame_energy > max(
        peak_energy * 0.001,
        1e-12
    )

    if np.any(active_frames):
        indexes = np.flatnonzero(active_frames)

        start = max(0, int(indexes[0]) - 2)
        end = min(len(active_frames), int(indexes[-1]) + 3)

        log_mel = log_mel[:, start:end]
        frame_energy = frame_energy[start:end]

    # Нормализация спектра.
    log_mel -= log_mel.mean(axis=0, keepdims=True)

    mfcc = dct(
        log_mel,
        type=2,
        axis=0,
        norm="ortho"
    )[1:N_MFCC + 1]

    # Нормализуем каждый временной кадр.
    norms = np.linalg.norm(mfcc, axis=0, keepdims=True)
    mfcc = mfcc / (norms + 1e-8)

    # Сохраняем тихие кадры как нулевые.
    local_peak = max(float(frame_energy.max()), 1e-12)
    quiet_frames = frame_energy < local_peak * 0.001

    mfcc[:, quiet_frames] = 0.0

    return mfcc.T.astype(np.float32)


def compare_features_dtw(reference, recording):
    reference_length = len(reference)
    recording_length = len(recording)

    if reference_length < 3 or recording_length < 3:
        return 1.0

    if recording_length < int(reference_length * 0.45):
        return 1.0

    # Косинусное расстояние между кадрами.
    similarities = reference @ recording.T
    costs = 1.0 - np.clip(similarities, -1.0, 1.0)

    reference_silent = np.linalg.norm(reference, axis=1) < 1e-6
    recording_silent = np.linalg.norm(recording, axis=1) < 1e-6

    both_silent = np.outer(
        reference_silent,
        recording_silent
    )

    one_silent = np.logical_xor(
        reference_silent[:, None],
        recording_silent[None, :]
    )

    costs[both_silent] = 0.0
    costs[one_silent] = 0.85

    # Subsequence DTW.
    # Эталон может находиться в любом месте записи.
    previous = np.zeros(
        recording_length + 1,
        dtype=np.float32
    )

    previous_lengths = np.zeros(
        recording_length + 1,
        dtype=np.int32
    )

    warp_penalty = 0.015

    for i in range(1, reference_length + 1):
        current = np.full(
            recording_length + 1,
            np.inf,
            dtype=np.float32
        )

        current_lengths = np.zeros(
            recording_length + 1,
            dtype=np.int32
        )

        for j in range(1, recording_length + 1):
            options = (
                previous[j - 1],
                previous[j] + warp_penalty,
                current[j - 1] + warp_penalty
            )

            best = int(np.argmin(options))
            frame_cost = costs[i - 1, j - 1]

            if best == 0:
                current[j] = options[0] + frame_cost
                current_lengths[j] = previous_lengths[j - 1] + 1

            elif best == 1:
                current[j] = options[1] + frame_cost
                current_lengths[j] = previous_lengths[j] + 1

            else:
                current[j] = options[2] + frame_cost
                current_lengths[j] = current_lengths[j - 1] + 1

        previous = current
        previous_lengths = current_lengths

    scores = previous[1:] / np.maximum(
        previous_lengths[1:],
        1
    )

    return float(np.min(scores))


@lru_cache(maxsize=1)
def load_references():
    references = []

    sound_files = sorted(
        SOUNDS_PATH.glob("*.ogg"),
        key=lambda path: int(path.stem)
    )

    for sound_file in sound_files:
        try:
            audio, sample_rate = load_audio(sound_file)

            features = extract_features(
                audio,
                sample_rate
            )

            if len(features) == 0:
                print(f"Пустой эталон: {sound_file.name}")
                continue

            references.append((
                int(sound_file.stem),
                features
            ))

        except Exception as error:
            print(f"Ошибка {sound_file.name}: {error}")

    print(f"Загружено эталонов: {len(references)}")

    return references


def recognize_audio_data(recording, recording_rate):
    recording = np.asarray(
        recording,
        dtype=np.float32
    )

    if recording.ndim == 2:
        recording = recording.mean(axis=1)

    if recording.size == 0:
        return []

    recording_features = extract_features(
        recording,
        int(recording_rate)
    )

    if len(recording_features) == 0:
        return []

    references = load_references()
    results = []

    for sound_id, reference_features in references:
        distance = compare_features_dtw(
            reference_features,
            recording_features
        )

        similarity = math.exp(
            -distance / DISTANCE_SCALE
        )

        results.append({
            "sound_id": sound_id,
            "similarity": similarity,
            "distance": distance
        })

    results.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return [
        {
            "sound_id": item["sound_id"],
            "similarity": round(item["similarity"], 4),
            "distance": round(item["distance"], 4)
        }
        for item in results[:5]
    ]


def recognize_audio(recording_path):
    recording, sample_rate = load_audio(recording_path)

    return recognize_audio_data(
        recording,
        sample_rate
    )


if __name__ == "__main__":
    recording_path = BASE_DIR / "test_audio.ogg"

    results = recognize_audio(recording_path)

    for result in results:
        print(
            f"Sound ID: {result['sound_id']} | "
            f"Similarity: {result['similarity'] * 100:.2f}% | "
            f"Distance: {result['distance']:.4f}"
        )
