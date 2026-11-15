"""Сценарий проекта: объединяет загрузку, модель, метрики и визуализацию."""
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil

import cv2
import numpy as np
import torch
import torchvision
from matplotlib import font_manager
from PIL import ImageFont

from easy_motion_data import VideoDataset, make_pairs
from easy_motion_models import PoseEstimator
from easy_motion_visualization.render import comparison_frame, plot_metrics, save_comparison
from easy_motion.metrics import compare_sequence, summarize


def run_project(reference_path, input_path, output_dir):
    """Обрабатывает оба режима и сохраняет результаты в output_dir."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg не найден в PATH. Инструкция установки находится в README.md")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = VideoDataset(reference_path)
    execution = VideoDataset(input_path)
    estimator = PoseEstimator()
    # В этом референсе слева манекен. Фильтруем центр рамки, не обрезая ногу спортсмена.
    ref_poses = estimator.predict_video(reference, min_center_x=0.4)
    my_poses = estimator.predict_video(execution)
    results = {}
    summaries = {}
    for mode in ("relative", "realtime"):
        pairs, fps = make_pairs(reference, execution, mode)
        records = compare_sequence(reference, execution, ref_poses, my_poses, pairs)
        results[mode] = records
        summaries[mode] = summarize(records)
        mode_dir = output_dir / mode
        mode_dir.mkdir(exist_ok=True)
        with (mode_dir / "frame_metrics.csv").open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=records[0].keys())
            writer.writeheader()
            for row in records:
                csv_row = {}
                for key, value in row.items():
                    # Неопределённые метрики записываем как пустые ячейки.
                    if isinstance(value, float) and not np.isfinite(value):
                        value = ""
                    csv_row[key] = value
                writer.writerow(csv_row)
        save_comparison(reference, execution, ref_poses, my_poses, records, mode, fps,
                        mode_dir / "comparison.mp4")
    plot_metrics(results, output_dir / "metrics.png")
    # Иллюстрация берётся из контрольного режима во время удара (около 78% интервала).
    demo_index = round(0.78 * (len(results["realtime"]) - 1))
    demo_row = results["realtime"][demo_index]
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 17)
    comparison_frame(reference, execution, ref_poses, my_poses, demo_row, "realtime", font).save(output_dir / "demo.png")
    input_hashes = {}
    for video in (reference, execution):
        input_hashes[video.path.name] = hashlib.sha256(video.path.read_bytes()).hexdigest()

    info = {
        "reference": reference.describe(),
        "execution": execution.describe(),
        "model": "Keypoint R-CNN ResNet-50 FPN",
        "weights": "COCO_V1",
        "person_threshold": 0.9,
        "point_threshold": 2.0,
        "minimum_common_points": 6,
        "reference_min_center_x": 0.4,
        "device": estimator.device,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "opencv": cv2.__version__,
        },
        "input_sha256": input_hashes,
        "detections": {
            "reference": sum(p["points"] is not None for p in ref_poses),
            "execution": sum(p["points"] is not None for p in my_poses),
        },
        "results": summaries,
    }
    summary_text = json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False)
    (output_dir / "summary.json").write_text(summary_text, encoding="utf-8")
    for mode, summary in summaries.items():
        print(f"{mode}: {summary['valid_pairs']}/{summary['pair_count']} пар; "
              f"d_cos={summary['cosine_distance']['mean']:.6f}; L1={summary['weighted_l1_px']['mean']:.3f} px")
    return info
