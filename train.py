from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(
    data="config.yaml",
    epochs=200,
    patience=10,
    resume=True,
    single_cls=True,
    max_det=1,
    batch=80,
)  # train the model
metrics = model.val(
    max_det=1
)  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format
