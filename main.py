from ultralytics import YOLO
import cv2
import os
import time
import json
import numpy as np
imagePath = 'images'
JSONPath = 'cls.json'
txtPath = 'label'
source = '' #视频源地址
# classificationPath = 'cls.txt'
key = 14 #鸟类索引,其它类型查看cls.txt
if os.path.exists(imagePath) == False:
    os.mkdir(imagePath)
if os.path.exists(txtPath) == False:
    os.mkdir(txtPath)
if os.path.exists(JSONPath) == True:
    os.remove(JSONPath)
# Load an official or custom model
model = YOLO('yolov5s.pt')  # Load an official Detect model

# Perform tracking with the model
results = model.track(source = source, stream=True)  # Tracking with default tracker
# xyxy = results.pandas().xyxy
for r in results:
    originSize = r.boxes.orig_shape
    
    names = r.names
    if r.boxes.cls.numel()>0:
        i=0
        temp = str(key) + str(time.time())
        tempTxtPaht = os.path.join(txtPath, temp+'.txt')
        txtImage = open(tempTxtPaht, 'w')
        for t in r.boxes.cls:
            className = r.boxes.cls[i].numpy()
            print(className, type(className))
            if className == np.array(key):
                x,y,w,h = r.boxes.xywhn[i].numpy()
                annotated_frame = r.plot()
                imageName = temp+'.jpg'
                targetPath = os.path.join(imagePath, imageName)
                cv2.imwrite(targetPath, annotated_frame)
                txtImage.write(str(x)+','+str(y)+','+str(w)+','+str(h)+','+str(originSize[0])+','+str(originSize[1]))
                txtImage.write('\n')
        txtImage.close()
                # f.write(json.dumps(tempnames))
                # f.write(',')
    # if os.path.exists(classificationPath) == False:
    #     txt = open(classificationPath, 'w')
    #     for index in names:
    #         txt.write(str(index)+':'+ names[index])
    #         txt.write('\n')
    # if r.boxes.cls.numel()>0:
    #     i=0
    #     for t in r.boxes.cls:
    #         x,y,w,h = r.boxes.xywhn[i].numpy()
    #         className = r.boxes.cls[i].numpy()
    #         print('1'+str(className), x,y,w,h)
    #         i = i + 1
        
    # xywh = r.boxes.xywh[0]
    # print(originSize, r.boxes.xywhn[0], r.boxes.cls[0])
    # annotated_frame = r.plot()
    # targetPath = os.path.join(imagePath, 'test'+str(index)+'.jpg')
    # cv2.imwrite(targetPath, annotated_frame)
    # index = index + 1
    # cv2.imshow("YOLOv8 Tracking", annotated_frame)
    # boxes = r.boxes
