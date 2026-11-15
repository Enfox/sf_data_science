"""Для коротких роликов храним кадры в памяти — так код проще читать."""
from pathlib import Path

import cv2
import numpy as np


class VideoDataset:
    """RGB-кадры одного видео, FPS и реальные временные метки."""

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Не найдено видео: {self.path}")
        cap = cv2.VideoCapture(str(self.path))
        self.frames = []
        times = []
        try:
            if not cap.isOpened():
                raise ValueError(f"Не удалось открыть: {self.path}")
            self.fps = float(cap.get(cv2.CAP_PROP_FPS))
            expected_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        finally:
            cap.release()
        if len(self.frames) < 2 or len(self.frames) != expected_count or self.fps <= 0:
            raise ValueError("Видео пустое, повреждено или содержит некорректный FPS")
        self.times = np.asarray(times)
        if not np.allclose(self.times, np.arange(len(self)) / self.fps, atol=0.002, rtol=0):
            raise ValueError("Этот учебный вариант рассчитан на видео с постоянным FPS")
        self.height, self.width = self.frames[0].shape[:2]
        self.duration = len(self) / self.fps

    def __len__(self):
        return len(self.frames)

    def describe(self):
        return {"file": self.path.name, "width": self.width, "height": self.height,
                "fps": self.fps, "frames": len(self), "duration_s": self.duration}


def nearest_frames(times, targets):
    """Находим ближайший реальный кадр к каждому заданному времени."""
    frame_ids = []
    for target_time in targets:
        # Для коротких видео можно просто проверить расстояние до каждого кадра.
        distances = np.abs(times - target_time)
        closest_frame = int(np.argmin(distances))
        frame_ids.append(closest_frame)
    return np.array(frame_ids, dtype=int)


def make_pairs(reference, execution, mode="relative", fps=30):
    """relative — одинаковая доля ролика; realtime — одинаковая секунда."""
    if fps <= 0:
        raise ValueError("FPS должен быть положительным")
    if mode == "relative":
        count = min(len(reference), len(execution),
                    round(max(reference.duration, execution.duration) * fps))
        if count < 2:
            raise ValueError("Нужно хотя бы две пары кадров")
        progress = np.linspace(0, 1, count)
        ref_ids = nearest_frames(reference.times, progress * reference.times[-1])
        my_ids = nearest_frames(execution.times, progress * execution.times[-1])
    elif mode == "realtime":
        fps = min(fps, reference.fps, execution.fps)
        times = np.arange(0, min(reference.duration, execution.duration) - 1e-9, 1 / fps)
        ref_ids = nearest_frames(reference.times, times)
        my_ids = nearest_frames(execution.times, times)
    else:
        raise ValueError("Режим должен быть relative или realtime")
    if not (np.all(np.diff(ref_ids) > 0) and np.all(np.diff(my_ids) > 0)):
        raise ValueError("Частота сравнения слишком высокая: кадры повторяются")
    return list(zip(ref_ids.tolist(), my_ids.tolist())), fps
