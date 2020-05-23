from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
import numpy as np
import argparse
import time
import cv2
import os
# Create your views here.
from django.http import HttpResponse

from django.conf import settings


def index(request):
    return render(request, 'parseImage/index.html')


def yoloScan(img, conf):
    # yolopath = os.path.join(settings.BASE_DIR, 'yolo-coco')
    yolopath = os.path.join(settings.BASE_DIR, 'yolo-fruits')
    threshold = 0.3
    # load the COCO class labels our YOLO model was trained on
    # labelsPath = os.path.sep.join([yolopath, "coco.names"])
    labelsPath = os.path.sep.join([yolopath, "obj.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")
    # derive the paths to the YOLO weights and model configuration
    # weightsPath = os.path.sep.join([yolopath, "yolov3.weights"])
    # configPath = os.path.sep.join([yolopath, "yolov3.cfg"])
    weightsPath = ch = os.path.sep.join([yolopath, "yolov3_custom_last.weights"])
    configPath = os.path.sep.join([yolopath, "yolov3_custom.cfg"])
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net=cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    # load our input image and grab its spatial dimensions
    image = cv2.imread(img)
    if image.size == 0:
        print("no img")
    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    # fruitlist = []
    fruitsfound = {}
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > conf:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    for val in classIDs:
        if val not in fruitsfound:
            fruitsfound[LABELS[val]] = classIDs.count(val)
    print(str(fruitsfound))
    print(str(classIDs))

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    ###
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf,
                            threshold)
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            #        # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
    # show the output image

    cv2.imwrite(os.path.join(settings.BASE_DIR, 'static/parseImage/' + img + '_mod.jpg'), image)
    # cv2.waitKey(0)
    return fruitsfound, img+'_mod.jpg'


def results(request):
    if request.method == 'POST' and request.FILES['fileup']:
        myfile = request.FILES['fileup']
        fs = FileSystemStorage()
        # conf = float(request.POST['confidence']) * 0.1
        conf = 0
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        fruitOut, outImg = yoloScan(filename, conf)
        # return HttpResponse("uploaded at %s.. the list of fruits are: %s" % (uploaded_file_url, str(fruitOut)))
        return render(request, 'parseImage/results.html', {
            'outImg': outImg,
            'fruitOut': fruitOut
        })
    # return render(request, 'parseImage/results.html')
    return HttpResponse("could not upload")
