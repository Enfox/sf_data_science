"""Keypoint R-CNN используется с готовыми весами; обучение не выполняется."""
import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    KeypointRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn,
)
from torchvision.transforms.functional import to_tensor


def legs_are_ambiguous(points):
    """Консервативная проверка слияния ног и почти исчезнувшего сегмента."""
    torso = np.linalg.norm(points[[5, 6]].mean(axis=0) - points[[11, 12]].mean(axis=0))
    knees = np.linalg.norm(points[13] - points[14])
    ankles = np.linalg.norm(points[15] - points[16])
    if knees < 0.2 * torso and ankles < 0.2 * torso:
        return True
    for hip, knee, ankle in [(11, 13, 15), (12, 14, 16)]:
        thigh = np.linalg.norm(points[hip] - points[knee])
        shin = np.linalg.norm(points[knee] - points[ankle])
        if min(thigh, shin) < 0.25 * max(thigh, shin):
            return True
    return False


class PoseEstimator:
    def __init__(self, person_threshold=0.9, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.person_threshold = person_threshold
        self.weights = KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
        self.names = list(self.weights.meta["keypoint_names"])
        self.model = keypointrcnn_resnet50_fpn(weights=self.weights, progress=False)
        self.model.to(self.device).eval()

    def _prediction(self, frame):
        with torch.inference_mode():
            prediction = self.model([to_tensor(frame).to(self.device)])[0]
        return {key: value.cpu().numpy() for key, value in prediction.items()}

    def _refine_legs(self, frame, original):
        """При слабой детекции пробуем повороты, затем возвращаем координаты назад."""
        best = original
        if original["point_scores"][11:17].min() >= 4.0 and not original["legs_ambiguous"]:
            return best
        points = original["points"]
        center = points[[5, 6, 11, 12]].mean(axis=0)
        torso = np.linalg.norm(points[[5, 6]].mean(axis=0) - points[[11, 12]].mean(axis=0))
        height, width = frame.shape[:2]
        for angle in (-40, -20, 20, 40):
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
            # Не используем угол, если исходная рамка спортсмена выйдет за кадр.
            x0, y0, x1, y1 = original["box"]
            corners = np.array([[x0, y0, 1], [x1, y0, 1],
                                [x1, y1, 1], [x0, y1, 1]]) @ matrix.T
            if (corners.min(axis=0) < 0).any() or (corners.max(axis=0) >= [width, height]).any():
                continue
            rotated = cv2.warpAffine(frame, matrix, (width, height))
            prediction = self._prediction(rotated)
            inverse = cv2.invertAffineTransform(matrix)
            candidates = []
            for i, score in enumerate(prediction["scores"]):
                if prediction["labels"][i] != 1 or score <= self.person_threshold:
                    continue
                xy = prediction["keypoints"][i, :, :2]
                mapped = np.column_stack([xy, np.ones(17)]) @ inverse.T
                distance = np.linalg.norm(mapped[[5, 6, 11, 12]].mean(axis=0) - center)
                if distance > torso:
                    continue
                scores = prediction["keypoints_scores"][i].astype(float)
                inside = ((mapped[:, 0] >= 0) & (mapped[:, 0] < width)
                          & (mapped[:, 1] >= 0) & (mapped[:, 1] < height))
                scores[~inside] = 0.0
                candidate = dict(original, points=mapped, point_scores=scores,
                                 person_score=float(score), rotation_deg=angle,
                                 legs_ambiguous=legs_are_ambiguous(mapped))
                candidates.append((distance, candidate))
            if candidates:
                # Сохраняем того же спортсмена, а не манекен рядом с ним.
                candidate = min(candidates, key=lambda item: item[0])[1]
                if candidate["legs_ambiguous"]:
                    continue
                if best["legs_ambiguous"] or candidate["point_scores"][11:17].min() > best["point_scores"][11:17].min():
                    best = candidate
        return best

    def predict(self, frame, min_center_x=0.0):
        """min_center_x — граница центра рамки как доля ширины изображения."""
        prediction = self._prediction(frame)

        # По очереди проверяем людей и выбираем самого уверенно найденного.
        selected = None
        best_score = self.person_threshold
        candidate_count = 0
        eligible_count = 0
        for i, score in enumerate(prediction["scores"]):
            is_person = prediction["labels"][i] == 1
            is_confident = score > self.person_threshold
            if not is_person or not is_confident:
                continue
            candidate_count += 1
            box = prediction["boxes"][i]
            center_x = (box[0] + box[2]) / (2 * frame.shape[1])
            is_in_area = center_x >= min_center_x
            if not is_in_area:
                continue
            eligible_count += 1
            if score > best_score:
                selected = i
                best_score = score

        result = {"points": None, "point_scores": None, "box": None,
                  "person_score": None, "candidate_count": candidate_count,
                  "eligible_count": eligible_count, "person_id": selected,
                  "rotation_deg": 0, "legs_ambiguous": False}
        if selected is not None:
            result["points"] = prediction["keypoints"][selected, :, :2].astype(float)
            result["point_scores"] = prediction["keypoints_scores"][selected].astype(float)
            result["box"] = prediction["boxes"][selected].astype(float)
            result["person_score"] = float(prediction["scores"][selected])
            result["legs_ambiguous"] = legs_are_ambiguous(result["points"])
            result = self._refine_legs(frame, result)
        return result

    def predict_video(self, video, min_center_x=0.0):
        poses = []
        for i, frame in enumerate(video.frames):
            poses.append(self.predict(frame, min_center_x))
            if (i + 1) % 60 == 0 or i + 1 == len(video):
                print(f"{video.path.name}: {i + 1}/{len(video)} кадров", flush=True)
        return poses
