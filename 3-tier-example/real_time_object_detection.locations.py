"""This program based on Adrian Rosebrock, Day 16 of 17, detects people and lists their positions in the frame"""
# T. Harris 8-May-2020
# T. Harris 16-May-2020 night -- scaled confidence by 100, added number of frames
# T. Harris 1-Jul-2020 count how many pairs of dancers there are (a dancer can be in more than one pair)
# T. Harris 9-Jan-2021 night -- remove the pairs detection attempt, prepare for refactoring to 3-tier model

# 
# Command line:  python real_time_object_detection.locations.py \
#    --prototxt MobileNetSSD_deploy.prototxt.txt \
#    --model MobileNetSSD_deploy.caffemodel \
#    --video <filename of mp4 video file>
# Or one one line for copy/paste:
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --video <filename of mp4 video file>
# Use savefrom.net to get YouTube video
# Use online-video-cutter.com
# Imports
# import the necessary packages

##from imutils.video import VideoStream # TH this would be for camera input
from imutils.video import FileVideoStream # TH added for reading video file
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

import math

# Main Program

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=20,
    help="minimum probability to filter weak detections")
ap.add_argument("-v", "--video", required=True,
    help="path to input video file") # TH added for video file
ap.add_argument("-f", "--frames", type=int, default=24,
    help="path to input video file") # TH added for video file
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class

# TH Color constants RGB
BLACK = (   0,   0,   0)
RED   = (   0,   0, 255)
BLUE =  (   255, 0,   0)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the camera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = FileVideoStream(args["video"]).start() # TH instead for video file
time.sleep(2.0) # warmup file open
fps = FPS().start() # start video timing measurements

# TH added
first_frame = True

# loop over the frames from the video stream
nFrames = args["frames"]

##while True:
for nFrame in range(nFrames):
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    # TH ensure that frame has content:
    if frame is None:
            break

    frame = imutils.resize(frame, width=400)
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # TH scale and trim the results
    scaleDetections = detections * np.array([1, 1, 100, w, h, w, h])
    boolSelectNonempty = ~np.all(scaleDetections == 0, axis=3)
    scaleDetectionsNonempty = scaleDetections[boolSelectNonempty]
    numDetections = scaleDetectionsNonempty.shape[0]

    # TH find aggregate properties
    if numDetections > 0:
            xMin = scaleDetectionsNonempty[:, [3, 5]].min().astype("int")
            yMin = scaleDetectionsNonempty[:, [4, 6]].min().astype("int")
            xMax = scaleDetectionsNonempty[:, [3, 5]].max().astype("int")
            yMax = scaleDetectionsNonempty[:, [4, 6]].max().astype("int")

            xCenters = scaleDetectionsNonempty[:, [3, 5]].mean(axis = 1).astype("int")
            yCenters = scaleDetectionsNonempty[:, [4, 6]].mean(axis = 1).astype("int")
            detectionCenters = np.column_stack((xCenters, yCenters))

            area = (xMax - xMin) * (yMax - yMin)

            meanDimension = math.sqrt(area)
            areaPercentOfFrame = int( ( area / (w * h)) * 100 ) # as percentage of frame for nice display
    
    # TH print aggregate data on console
    print("frame #: {0}".format(nFrame))
    print("Identified {0} dancers".format(numDetections))
    print("They form a rectangle with area {4} and these corners: ({0},{1}), ({2},{3})".format(xMin, yMin, xMax, yMax, area))

    # loop over the detections
    
    for nDetection in np.arange(0, numDetections):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = scaleDetectionsNonempty[nDetection, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            # TH adjusted for non-empty rows array
            goodDetection = scaleDetectionsNonempty[nDetection]
            box = goodDetection[3:7]
            (startX, startY, endX, endY) = box.astype("int")
            # draw the prediction on the frame
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                BLUE, 2)
            # TH draw box of detected group on the frame
            cv2.rectangle(frame, (xMin, yMin), (xMax, yMax),
                RED, 2)            # TH write the aggregate results on the frame
            text = "{} % of frame {}: contains {} detections ".format(areaPercentOfFrame, nFrame, numDetections)
            cv2.putText(frame, text, (0, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
##cv2.destroyAllWindows()
vs.stop()
