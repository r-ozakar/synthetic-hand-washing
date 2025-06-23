from ultralytics import YOLO

# Load a model
model = YOLO("./yolov8n-seg.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="./yaml/training1.yaml", epochs=5, flipud=0.0, fliplr=0.0)
model.export()