"""Визуализация отделена от расчётов: рисунки не меняют исходные позы."""
from pathlib import Path
import shutil
import subprocess

import cv2
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BLUE, ORANGE = (36, 103, 200), (212, 109, 31)
LIMBS = [(2, 0), (2, 4), (1, 0), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9),
         (12, 14), (14, 16), (11, 13), (13, 15), (6, 5), (12, 11), (6, 12), (5, 11)]
MODE_LABELS = {"relative": "Полные ролики: относительное время",
               "realtime": "Общий интервал: без изменения темпа"}


def draw_pose(frame, pose, color, threshold=2.0):
    image = frame.copy()
    if pose["points"] is None:
        return image
    points = pose["points"]
    visible = (pose["point_scores"] > threshold) & np.isfinite(points).all(axis=1)
    width = max(2, round(frame.shape[0] / 300))
    for a, b in LIMBS:
        if visible[a] and visible[b]:
            start, end = [tuple(np.rint(p).astype(int)) for p in points[[a, b]]]
            cv2.line(image, start, end, color, width, cv2.LINE_AA)
    for point in points[visible]:
        cv2.circle(image, tuple(np.rint(point).astype(int)), width + 1, color, -1, cv2.LINE_AA)
    return image


def comparison_frame(reference, execution, ref_poses, my_poses, row, mode, font):
    image = Image.new("RGB", (720, 800), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    for left, video, poses, frame_id, color in [
        (0, reference, ref_poses, row["reference_frame"], BLUE),
        (360, execution, my_poses, row["input_frame"], ORANGE),
    ]:
        frame = Image.fromarray(draw_pose(video.frames[frame_id], poses[frame_id], color))
        frame.thumbnail((360, 640), Image.Resampling.LANCZOS)
        image.paste(frame, (left + (360-frame.width)//2, 48 + (640-frame.height)//2))
    draw.text((12, 14), f"Референс: {row['reference_time_s']:.3f} с", font=font, fill=BLUE)
    draw.text((372, 14), f"Повторение: {row['input_time_s']:.3f} с", font=font, fill=ORANGE)
    text = (f"d_cos: {row['cosine_distance']:.5f}  |  L1: {row['weighted_l1_px']:.2f} px"
            if row["status"] == "ok" else f"Нет оценки: {row['status']}")
    draw.text((12, 700), text + "  |  меньше — ближе", font=font, fill=(25, 35, 50))
    draw.text((12, 735), MODE_LABELS[mode], font=font, fill=(25, 35, 50))
    draw.text((12, 770), f"Общих точек: {row['common_points']}/17 · Метрики после совмещения",
              font=font, fill=(25, 35, 50))
    return image


def save_comparison(reference, execution, ref_poses, my_poses, records, mode, fps, path):
    """OpenCV пишет кадры; FFmpeg переводит MP4 в совместимый с браузерами H.264."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Установите FFmpeg и добавьте его в PATH (см. README.md)")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (720, 800))
    if not writer.isOpened():
        raise RuntimeError("Не удалось создать видео")
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 17)
    try:
        for row in records:
            image = comparison_frame(reference, execution, ref_poses, my_poses, row, mode, font)
            writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(temporary),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                   check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    temporary.unlink()


def plot_metrics(results, path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col", sharey="row", constrained_layout=True)
    for col, (mode, records) in enumerate(results.items()):
        x = np.linspace(0, 100, len(records)) if mode == "relative" else [r["reference_time_s"] for r in records]
        axes[0, col].plot(x, [r["cosine_distance"] for r in records], color="#2467C8")
        axes[1, col].plot(x, [r["weighted_l1_px"] for r in records], color="#D46D1F")
        axes[0, col].set_title(MODE_LABELS[mode], fontsize=11)
        axes[1, col].set_xlabel("Доля ролика, %" if mode == "relative" else "Время, с")
        for ax in axes[:, col]:
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.15)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Косинусное расстояние")
    axes[1, 0].set_ylabel("Взвешенное L1, px референса")
    fig.suptitle("Сравнение поз при двух способах сопоставления времени")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
