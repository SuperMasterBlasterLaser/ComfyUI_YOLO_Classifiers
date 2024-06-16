import os

import folder_paths
from torchvision import transforms
from ultralytics import YOLO

model_path = folder_paths.models_dir
folder_paths.add_model_folder_path("ultralytics_classifiers", os.path.join(model_path, "ultralytics", "classifiers"))


class YOLOClassifierModelLoader:
    # Download these models from https://docs.ultralytics.com/tasks/classify/
    # Place them on models/ultralytics/classification folder
    model_names = (
        'yolov8n-cls.pt',
        'yolov8s-cls.pt',
        'yolov8m-cls.pt',
        'yolov8l-cls.pt',
        'yolov8x-cls.pt'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (cls.model_names,)
            }
        }

    RETURN_TYPES = ("YOLO_CLASSIFIER_MODEL",)
    RETURN_NAMES = ("YOLO classifier model",)
    FUNCTION = "load_model"
    CATEGORY = "SuperMasterBlasterLaser/ComfyUI_YOLO_Classifiers"

    def load_model(self, model_name):
        model_path = folder_paths.get_full_path("ultralytics_classifiers", model_name)

        model = YOLO(model_path)
        return (model,)


class YOLOClassify:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "yolo_classifier_model": ("YOLO_CLASSIFIER_MODEL",),
                "image_to_classify": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Class name with highest score",)
    FUNCTION = "classify"
    CATEGORY = "SuperMasterBlasterLaser/ComfyUI_YOLO_Classifiers"

    def classify(self, yolo_classifier_model, image_to_classify):
        image_to_classify = image_to_classify.permute(0, 3, 1, 2)

        if yolo_classifier_model.fp16:
            image_to_classify = image_to_classify.half()
        else:
            image_to_classify = image_to_classify.float()

        image_to_classify = image_to_classify.squeeze(0)
        transform = transforms.Compose([
            transforms.Resize(size=(224, 224))
        ])

        img = transform(image_to_classify).unsqueeze(0)
        result = yolo_classifier_model(img)

        class_name = None
        for r in result:
            top1 = r.probs.top1
            class_name = r.names[top1]
            break

        return (class_name,)
