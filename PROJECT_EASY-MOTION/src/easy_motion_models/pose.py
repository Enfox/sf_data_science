"""Keypoint R-CNN используется с готовыми весами; обучение не выполняется."""
import torch
from torchvision.models.detection import (
    KeypointRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn,
)
from torchvision.transforms.functional import to_tensor


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

    def predict(self, frame, min_center_x=0.0):
        """min_center_x — граница центра рамки как доля ширины изображения."""
        with torch.inference_mode():
            prediction = self.model([to_tensor(frame).to(self.device)])[0]
        for key, value in prediction.items():
            prediction[key] = value.cpu().numpy()

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
                  "eligible_count": eligible_count}
        if selected is not None:
            result["points"] = prediction["keypoints"][selected, :, :2].astype(float)
            result["point_scores"] = prediction["keypoints_scores"][selected].astype(float)
            result["box"] = prediction["boxes"][selected].astype(float)
            result["person_score"] = float(prediction["scores"][selected])
        return result

    def predict_video(self, video, min_center_x=0.0):
        poses = []
        for i, frame in enumerate(video.frames):
            poses.append(self.predict(frame, min_center_x))
            if (i + 1) % 60 == 0 or i + 1 == len(video):
                print(f"{video.path.name}: {i + 1}/{len(video)} кадров", flush=True)
        return poses
