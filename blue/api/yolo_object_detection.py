import cv2
import numpy as np
import os

path=os.path.abspath(__file__)
str_path=path.split("/")[0:-1]
str_path="/".join(str_path)
weights=str_path+"/"+"yolov3.weights"
cfg=str_path+"/"+"yolov3.cfg"
coconames=str_path+"/"+"coco.names"


# Load Yolo
net = cv2.dnn.readNet(weights,cfg)
classes = []
with open(coconames, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))



def yolo_return_names(imgarr):
    print("Start")
    #npimg = np.fromstring(img, np.uint8)
    #img = cv2.imdecode(img,cv2.IMREAD_COLOR)
    img=cv2.resize(imgarr,(416,416))
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    l=[]
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            l.append(label)
    return l
