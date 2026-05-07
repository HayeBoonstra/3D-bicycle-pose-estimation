import os

# Disable optional inference sub-models we are not using in this script.
os.environ.setdefault("CORE_MODEL_SAM_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_SAM3_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_GAZE_ENABLED", "False")

import supervision as sv
from inference import get_model
from PIL import Image

image = Image.open("/home/hayepc/3D-bicycle-pose-estimation/data/bicycle_pose_dataset/images/train/clip_docks_scene_27911967/frame_0057.png")

print("Loading model...")
model = get_model("rfdetr-2xlarge")

print("Inferring...")
predictions = model.infer(image, confidence=0.5)[0]

print("Annotating image...")
bicycle_predictions = [
    prediction
    for prediction in predictions.predictions
    if prediction.class_name.lower() == "bicycle"
]

labels = [prediction.class_name for prediction in bicycle_predictions]
bicycle_detections = sv.Detections.from_inference(
    predictions.model_copy(update={"predictions": bicycle_predictions})
)

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, bicycle_detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, bicycle_detections, labels)

output_path = os.path.dirname(__file__) + "/annotated_output.jpg"
annotated_image.save(output_path)
print(f"Saved annotated image to {output_path}")