
import io
import os
import base64
import json
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime.backend as backend
import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Annotated
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse


app = FastAPI()
model_path = "models/yolov5s_dolly.onnx"
session = ort.InferenceSession(model_path)

# Model listing endpoint
# A list of available models
available_models = ["yolov5s_dolly", "fasterRCNN_dolly"]
available_classes= {0:'Dolly',1:"Wheel"}


@app.get("/get_model_names")
async def get_model_names():
	return {"Models": available_models}

# Model labels endpoint

@app.get("/get_model_labels/{model_name}")
async def get_model_labels(model_name: str):
    # Return output labels for a specific model
    if (model_name=='fasterRCNN_dolly'):
        return {"labels": ["background", "dolly", "wheel"]}
    elif (model_name=="yolov5s_dolly"):
      return {"labels": ["dolly", "wheel"]}
    else:
      raise HTTPException(status_code=404, detail="Model not found")
  
#Predict bounding boxes using Faster RCNN
def predict_bboxes_with_faster_rcnn(image_np,confidence_threshold=0.5):
    def preprocess(image):

        transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        ])

        # Apply the transformation to the image
        tensor = transform(image)


        # Add an extra dimension to the tensor to represent the batch size
        tensor = tensor.unsqueeze(0)
        return tensor

    # First load the onnx model
    if (os.path.exists(r'models/faster_rcnn_dolly.onnx')):
        print('Model exists')
    onnx_model = onnx.load(r'models/faster_rcnn_dolly.onnx')
    onnx.checker.check_model(onnx_model)


    # compute ONNX Runtime output prediction
    tensor = preprocess(Image.fromarray(image_np))
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    sess_options = ort.SessionOptions()
    # Below is for optimizing performance
    sess_options.intra_op_num_threads = 24
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    #1st method
    # ort_session = ort.InferenceSession("models/faster_rcnn_dolly.onnx", sess_options=sess_options)
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(tensor)}
    # ort_outs = ort_session.run(None, ort_inputs)
    # print(ort_outs)
    
    #2nd method
    rep = backend.prepare(onnx_model, 'CPU')

    outputs= rep.run(to_numpy(tensor))
    print(outputs)
    
    # Process outputs
    # Extract bounding boxes, class and confidence values
    boxes = outputs[0]
    labels = outputs[1]
    scores = outputs[2]
    bounding_boxes = []
    for box, label, score in zip(boxes, labels, scores):
        if score>confidence_threshold:
        
            xmin, ymin, xmax, ymax = box
            bounding_box = {"class": available_classes[int(label)-1], "accuracy": float(score), "xmin": int(xmin), "ymin": int(ymin), "xmax": int(xmax), "ymax": int(ymax)}
            bounding_boxes.append(bounding_box)
    return {"bounding_boxes": bounding_boxes}




#Predict bounding boxes using YOLOv5
def predict_bboxes_with_yolov5(image_np):
    #resize image
    print('loading model')
    model = torch.hub.load('yolov5', 'custom', path='models/yolov5s_dolly.onnx', source='local')
    print('running inference')
    results = model(image_np)  # inference

    # Process outputs
    df = results.pandas().xyxy[0]  # get first detected object
    bounding_boxes = []
    idx=0
    for label in df['class']:
        bounding_box = {
        "class": available_classes[int(label)],
        "accuracy": float(df["confidence"][idx]),
        "xmin": int(df["xmin"][idx]),
        "ymin": int(df["ymin"][idx]),
        "xmax": int(df["xmax"][idx]),
        "ymax": int(df["ymax"][idx])
        }
        idx=idx+1
        bounding_boxes.append(bounding_box)
    return {"bounding_boxes": bounding_boxes}


# Inference endpoint: base64 encoded image input
@app.post("/predict_base64")
def predict_base64(model_name: Annotated[str, Form()], image_base64: Annotated[str, Form()]):
    if model_name not in available_models:
        raise HTTPException(status_code=404, detail="Model not found")
    print('Decoding Image')
    image_data = base64.b64decode(image_base64)
    image_np = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_np = cv2.resize(image_np, (640, 640))
    print('running inference')
    # Run inference
    if(model_name =='yolov5s_dolly'):
        return predict_bboxes_with_yolov5(image_np)
    elif(model_name =='fasterRCNN_dolly'):
        return predict_bboxes_with_faster_rcnn(image_np)

@app.post("/predict_bounding_boxes")
async def predict_bounding_boxes(model_name: Annotated[str, Form()],confidence_threshold: Annotated[float, Form()],image: Annotated[UploadFile, File()] ):
    if model_name not in available_models:
        raise HTTPException(status_code=404, detail="Model not found")
    image_data = await image.read()
    image_np = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_np = cv2.resize(image_np, (640, 640))


    # Run inference
    if(model_name =='yolov5s_dolly'):
        return predict_bboxes_with_yolov5(image_np)
    elif(model_name =='fasterRCNN_dolly'):
        return predict_bboxes_with_faster_rcnn(image_np,confidence_threshold)
    else:
        raise HTTPException(status_code=404, detail="Model not found")

@app.post("/predict_image_with_bounding_boxes")
async def predict_image_with_bounding_boxes(model_name: Annotated[str, Form()],confidence_threshold: Annotated[float, Form()],file: Annotated[UploadFile, File()] ):
    # read the image from the request
    print('Decoding Image')
    image_data = await file.read()
    image_np = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_np = cv2.resize(image_np, (640, 640))
    print('Fetching bounding boxes')
    # perform inference to get bounding boxes
    # Run inference
    if(model_name =='yolov5s_dolly'):
        bboxes = predict_bboxes_with_yolov5(image_np)
    elif(model_name =='fasterRCNN_dolly'):
        bboxes = predict_bboxes_with_faster_rcnn(image_np,confidence_threshold)
    else:
        raise HTTPException(status_code=404, detail="Model not found")
    # draw bounding boxes on the image
    print('Plotting bounding boxes on image')
    image=np.array(image_np)
    image=image.astype(np.uint8)
    for box in bboxes['bounding_boxes']:
        class_name=box['class']
        accuracy=box['accuracy']
        xmin=box['xmin']
        ymin=box['ymin']
        xmax=box['xmax']
        ymax=box['ymax']
        color = (0, 255, 0) if class_name=='Dolly' else (0, 0, 255) # green for dolly, red for wheel
        thickness = 2
        print(f"Drawing box for {class_name} with accuracy {accuracy}")

        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
        image = cv2.putText(image, f"{class_name} ({accuracy:.2f})", (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    print('Returning image')
    # encode the image as jpeg or png, depending on the file type
    image_file_type = file.filename.split('.')[-1]
    if image_file_type == 'jpeg' or image_file_type == 'jpg':
        image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
    elif image_file_type == 'png':
        image_bytes = cv2.imencode('.png', image)[1].tobytes()
    else:
        raise HTTPException(status_code=400, detail="Unsupported File Type: only jpeg and png are supported")

    # return the image as a response
    return StreamingResponse(io.BytesIO(image_bytes), media_type=f"image/{image_file_type}")

#------------------------------------------------------------------------------------------------------------

