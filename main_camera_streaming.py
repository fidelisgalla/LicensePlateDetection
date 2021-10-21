#this file is the main

#general library
import cv2
import os
import time
import imutils
import argparse

#CRAFT library
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import numpy as np
import uuid


from imutils.video import VideoStream

#YOLO library
def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='License Plate Detection')
parser.add_argument('--ImgPath', default='img/img1a.jpg', type=str, help='folder to test area')
parser.add_argument('--YOLO_conf', default='D:/GIG/Project/MaskRCNNPlate/YOLOModel/yolov3.cfg', type=str, help='pretrained model YOLO')
parser.add_argument('--YOLO_weight', default='D:/GIG/Project/MaskRCNNPlate/YOLOModel/yolov3_custom_last(2).weights', type=str, help='pretrained model YOLO')
parser.add_argument('--YOLO_confidence', default=0.7, type=float, help='YOLO confidence detection')
parser.add_argument('--YOLO_ratio_detected', default=0.4, type=float, help='YOLO ratio ')
parser.add_argument('--Camera', default=True, type=str2bool, help='Use Camera For Image Aquistion')

parser.add_argument('--CRAFT_model', default='model/xxx, type=str', help='pretrained CRAFT model')

args = parser.parse_args()


def startingCamera():
    net = cv2.dnn.readNet(args.YOLO_weight, args.YOLO_conf)
    classes = ["plat"]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    #colors = np.random.uniform(0, 255, size=(len(classes), 3))

# initialize the camera and grab a reference to the raw camera capture
    if args.Camera:
        try:
            img = cv2.imread(args.ImgPath)
            print("[INFO] successful loading the image...")
            img = cv2.resize(img, (300, 300))  # do we need to do this?
        except:
            print("[INFO] failed loading the image...")
    else:
            #try to get frame by frame from the video stream
        print("[INFO] starting video stream...")
        grab,vs = cv2.VideoCapture(0)
        time.sleep(2.0)

        while True:
            frame = vs.read()
            img = cv2.resize(frame,(300,300))
            height, width = img.shape[:2]
            if not grab:
                break
            else:
                print("[INFO] failed connecting to video stream...")

            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

    #Showing informations on the screen
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

                        boxes.append([x, y, w, h])  # boxes is the coordinate of the detected object
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            font = cv2.FONT_HERSHEY_PLAIN

    #calculate the area
            AreaOfImage = height*width
            AreaOfDetected = w*h

        if AreaOfDetected/AreaOfImage >=args.YOLO_ratio_detected:
            print('Area Berhasil Dideteksi dan Gambar Disimpan')
            temp_folder = './temp/'
            if not os.path.isdir(temp_folder):
                os.mkdir(temp_folder)

            path = "{base_path}\{rand}{ext}".format(base_path=temp_folder,rand=str(uuid.uuid4()), ext=".jpg")
            cropped = img[x:x+w,y:y+h]

        cv2.imwrite(path,cropped)


        return cropped

if __name__ == '__main__':
    startingCamera()


