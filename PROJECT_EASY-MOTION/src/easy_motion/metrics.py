"""Метрики не зависят от способа загрузки видео или архитектуры нейросети."""
import numpy as np


def cosine_distance(first, second):
    """1 − косинусное сходство. Чем меньше, тем ближе направления векторов."""
    first = np.asarray(first, dtype=float).ravel()
    second = np.asarray(second, dtype=float).ravel()
    if first.shape != second.shape or not first.size:
        raise ValueError("Нужны непустые векторы одинакового размера")
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if not np.isfinite(denominator) or denominator <= np.finfo(float).eps:
        raise ValueError("Косинус не определён для нулевого или неконечного вектора")
    return float(1 - np.clip(np.dot(first, second) / denominator, -1, 1))


def weighted_distance(first, second, weights):
    """Средняя взвешенная L1-ошибка по суставам, в пикселях референса."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if first.ndim != 2 or first.shape[1] != 2 or first.shape != second.shape or weights.shape != (len(first),):
        raise ValueError("Ожидаются координаты (N, 2) и N весов")
    if not all(np.isfinite(value).all() for value in (first, second, weights)):
        raise ValueError("Координаты и веса должны быть конечными")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Веса должны быть неотрицательными, их сумма — положительной")
    errors = np.abs(first - second).sum(axis=1)
    return float(np.dot(errors, weights) / weights.sum())


def align_pose(execution, reference):
    """Подбираем аффинную матрицу: [x, y, 1] @ A ≈ [x_ref, y_ref]."""
    X = np.column_stack([execution, np.ones(len(execution))])
    matrix, _, rank, _ = np.linalg.lstsq(X, reference, rcond=None)
    if rank < 3:
        raise ValueError("Точки лежат на одной прямой: совмещение неоднозначно")
    return X @ matrix, matrix


def compare_pose(reference, execution, point_threshold=2.0, min_points=6):
    result = {"status": "ok", "common_points": 0, "cosine_distance": np.nan,
              "weighted_l1_px": np.nan, "centered_cosine_distance": np.nan,
              "affine_determinant": np.nan}
    if reference["points"] is None or execution["points"] is None:
        result["status"] = "missing_person"
        return result
    mask = ((reference["point_scores"] > point_threshold)
            & (execution["point_scores"] > point_threshold)
            & np.isfinite(reference["points"]).all(axis=1)
            & np.isfinite(execution["points"]).all(axis=1))
    result["common_points"] = int(mask.sum())
    if mask.sum() < min_points:
        result["status"] = "insufficient_points"
        return result
    # Для сравнения удара нужны обе ноги: бедро, колено и голеностоп.
    if not mask[11:17].all() or reference.get("legs_ambiguous", False) or execution.get("legs_ambiguous", False):
        result["status"] = "unreliable_legs"
        return result
    try:
        ref_points, my_points = reference["points"][mask], execution["points"][mask]
        aligned, matrix = align_pose(my_points, ref_points)
        result.update(
            cosine_distance=cosine_distance(ref_points, aligned),
            weighted_l1_px=weighted_distance(ref_points, aligned, execution["point_scores"][mask]),
            centered_cosine_distance=cosine_distance(ref_points-ref_points.mean(0), aligned-aligned.mean(0)),
            affine_determinant=float(np.linalg.det(matrix[:2])),
        )
    except (ValueError, np.linalg.LinAlgError):
        result["status"] = "invalid_geometry"
    return result


def compare_sequence(reference, execution, ref_poses, my_poses, pairs):
    records = []
    for i, (ref_id, my_id) in enumerate(pairs):
        record = {"pair_index": i, "reference_frame": ref_id, "input_frame": my_id,
                  "reference_time_s": float(reference.times[ref_id]),
                  "input_time_s": float(execution.times[my_id])}
        record.update(compare_pose(ref_poses[ref_id], my_poses[my_id]))
        records.append(record)
    return records


def summarize(records):
    valid = [row for row in records if row["status"] == "ok"]
    if not valid:
        raise ValueError("Нет пригодных пар кадров: проверьте детекции и пороги")
    summary = {"pair_count": len(records), "valid_pairs": len(valid),
               "reflected_pairs": sum(row["affine_determinant"] < 0 for row in valid)}
    summary["skipped_pairs"] = {}
    for row in records:
        if row["status"] != "ok":
            reason = row["status"]
            summary["skipped_pairs"][reason] = summary["skipped_pairs"].get(reason, 0) + 1
    for name in ("cosine_distance", "weighted_l1_px", "centered_cosine_distance"):
        values = [row[name] for row in valid]
        summary[name] = {"mean": float(np.mean(values)), "median": float(np.median(values)),
                         "min": float(np.min(values)), "max": float(np.max(values))}
    return summary
