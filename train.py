from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(
    data="config.yaml",
    epochs=100,
    patience=10,
    resume=True,
    single_cls=True,
    device="mps",
    max_det=1,
    val=False,
    batch=-1,
)  # train the model
metrics = model.val(
    max_det=1, device="mps"
)  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format
