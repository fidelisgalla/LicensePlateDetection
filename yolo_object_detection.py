import cv2
import numpy as np
import glob
import random
import string
import os
import argparse
from matplotlib import pyplot as plt
import multiprocessing
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from CRAFTTextDetection import CraftText
from utils.ArrayChange import ArrayModificationOrder
from utils.CharacterDetection import histogram_of_pixel_projection
from utils.FourPointTransform import four_point_transform


#argument declaration
def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def splitpath(path):
    path = os.path.normpath(path)
    return path.split(os.sep)

parser = argparse.ArgumentParser(description='License Plate Detection')
parser.add_argument('--ImgPath', default='img/img8.jpg', type=str, help='folder to test area')
parser.add_argument('--YOLO_conf', default='D:/GIG/Project/MaskRCNNPlate/YOLOModel/yolov3.cfg', type=str, help='pretrained model YOLO')
parser.add_argument('--YOLO_weight', default='D:/GIG/Project/MaskRCNNPlate/YOLOModel/yolov3_custom_last (2).weights', type=str, help='pretrained model YOLO')
parser.add_argument('--YOLO_confidence', default=0.7, type=float, help='YOLO confidence detection')
parser.add_argument('--YOLO_ratio_detected', default=0.01, type=float, help='YOLO ratio ')
parser.add_argument('--YOLO_result_folder', default='ResultYOLO', type=str, help='folder to save the YOLO image detection')
parser.add_argument('--CRAFT_model', default='model/xxx, type=str', help='pretrained CRAFT model')

args = parser.parse_args()



def YOLOLicensePlateImageDetection():
    net = cv2.dnn.readNet(args.YOLO_weight,args.YOLO_conf)
    classes = ["License_Plate"]
    images_path = glob.glob(args.ImgPath)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    random.shuffle(images_path)
# loop through all the images
    for img_path in images_path:
    # Loading image
        imgOrigin = cv2.imread(img_path)
        img = imgOrigin
        #img = cv2.resize(img, None, fx=0.2, fy=0.2)
        height, width, channels = img.shape

    # Detecting objects
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
                if confidence > 0.3:
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
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                cropped = img[y:y+h,x:x+w]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)
            #cv2.putText(img, text, (x, y + 30), font, 2, color, 3)

        #it should be another function in a class

        AreaOfImage = height * width
        AreaOfDetected = w * h
        if AreaOfDetected/AreaOfImage <=args.YOLO_ratio_detected:
            break
        resultFolderYOLO = args.YOLO_result_folder
        if not os.path.isdir(resultFolderYOLO):
            os.mkdir(resultFolderYOLO)
        path = "{base_path}\{rand}{ext}".format(base_path=resultFolderYOLO, rand=''.join(random.choice(string.ascii_uppercase + string.digits)
                                                                                    for _ in range(16)), ext=".jpg")
        cropped = img[y:y + h, x:x + w]
        cv2.imwrite(path,cropped)

    cv2.destroyAllWindows()

    return cropped,path, imgOrigin


if __name__ == '__main__':

    YOLOResultFolder = args.YOLO_result_folder
    croppedImageYOLODetection = YOLOLicensePlateImageDetection()[0]
    pathImageYOLODetection = os.path.abspath(YOLOLicensePlateImageDetection()[1])
    if os.path.isfile(pathImageYOLODetection):  #bisa digantikan dengan with open
        craft = CraftText.load_data(pathImageYOLODetection)

    pathCraftDetection = 'resultCRAFT/res_'+ str(splitpath(YOLOLicensePlateImageDetection()[1])[1])
    #array modification is used to modify the order and add some padding
    coordinateImageChanged = ArrayModificationOrder(craft)
    newCoordinateAfterMofication = coordinateImageChanged.newArray()

    x_CoordinateAfterCraft,y_CoordinateAfterCraft = int(newCoordinateAfterMofication[0][0]),int(newCoordinateAfterMofication[0][1])
    h_CoordinateAfterCraft,w_CoordinateAfterCraft = int((newCoordinateAfterMofication[2][1]-newCoordinateAfterMofication[0][1])),\
            int((newCoordinateAfterMofication[1][0]-newCoordinateAfterMofication[0][0]))


    #slice the image based on new array
    croppedFrameCraftDetection = croppedImageYOLODetection[y_CoordinateAfterCraft:y_CoordinateAfterCraft+h_CoordinateAfterCraft,
                                 x_CoordinateAfterCraft:x_CoordinateAfterCraft+w_CoordinateAfterCraft]

    #writing the image as a result of craft modification
    folderAfterCraftModificationDetection = 'ResultCRAFTAndModified'
    if not os.path.isdir(folderAfterCraftModificationDetection):
        os.mkdir(folderAfterCraftModificationDetection)

    cv2.imwrite(folderAfterCraftModificationDetection+'/'+str(splitpath(YOLOLicensePlateImageDetection()[1])[1]),
                croppedFrameCraftDetection)

    #four point transform
    # top-left, top-right, bottom-right, and bottom-left

    fourPointTransform = four_point_transform(croppedImageYOLODetection,
                                              newCoordinateAfterMofication)

    #character detection
    imageHorizontalProjectionAfterFourPoint = histogram_of_pixel_projection(fourPointTransform)
    imageHorizontalProjectionBeforeFourPoint = histogram_of_pixel_projection(croppedFrameCraftDetection)

    listofCharactersRecognized = imageHorizontalProjectionAfterFourPoint[0]
    #for i,image in enumerate(listofCharactersRecognized):
     #   cv2.imwrite('ResultCRAFTAndModified/imageCharacter_'+str(i),image)

    custom_config = r'--oem 3 --psm 6'
    textTesseract = pytesseract.image_to_string(listofCharactersRecognized[1],config = custom_config)
    print(textTesseract)
    cv2.imshow("frame",listofCharactersRecognized[1])
    key = cv2.waitKey(0)

    fig, ((ax1, ax2), (ax3, ax4),(ax5, ax6)) = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True)
    ax1.imshow(YOLOLicensePlateImageDetection()[2],aspect = 'auto')
    ax2.imshow(croppedImageYOLODetection,aspect = 'auto',interpolation = 'nearest')
    ax3.imshow(croppedFrameCraftDetection)
    ax4.imshow(fourPointTransform)
    ax5.imshow(imageHorizontalProjectionAfterFourPoint[1])
    ax6.imshow(imageHorizontalProjectionBeforeFourPoint[1])

    plt.show()
#after this the process is continued with four point perspective

#doing some recogntion with