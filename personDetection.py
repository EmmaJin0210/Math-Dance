"""
returns coords of dancers
"""

# Emma Jin   created: 2021/1/10
# Tom Harris modified: 2021/1/20:
#     add filename on command line
#     pretty printer for array debugging
#     use openCV contour feature functions to process HoG-detected group contour 
#
# --------------------------------------------------------------------
# tutorial used: https://thedatafrog.com/en/articles/human-detection-video/
# https://stackoverflow.com/questions/51074984/sorting-according-to-clockwise-point-coordinates/51075469
# https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates

import cv2
import numpy as np
from functools import reduce
import operator
import math
import argparse
import pprint

############################## PRESENTATION TIER ###############################
def showRects(frame, rects):
    for (p1,p2,p3,p4) in rects:
        cv2.rectangle(frame, (p1,p2), (p3,p4), (0,255,0), 2)
    return frame

##def showCoords(frame, coord):
##    frame = cv2.circle(frame, coord, radius=10, color=(0, 0, 255), thickness=2)
##    return frame

def showCoords(coord):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(coord)

def showOutline(frame, pLowerLeft, pUpperRight):
    """This function is SUPPOSED TO overlay group bounding box on image"""
    # T. Harris 21-Jan-2021 I think this function may have an error
    # Results I saw didn't look right; not sure
    
    print("min, max vertices: ", pLowerLeft, pUpperRight) # debug
    cv2.rectangle(frame, pLowerLeft, pUpperRight, (0, 255, 255), 20) # thick line
    
################################## DATA TIER ###################################
def getCoords(frame, hog):
    frame = cv2.resize(frame, (720, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = getRects(frame,hog)
##    frame = showRects(frame, rects)
    coords = np.array([[int(x+w/2), int(y+h/2)] for (x,y,w,h) in rects])
    return frame, coords

def getRects(frame, hog):
    rects, _ = hog.detectMultiScale(frame, winStride=(8,8))
    rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
    return rects
    
############################# APPLICATION TIER #################################
def calcNumDancers(coordlayer):
    return(len(coordlayer))

def calcBoundingBox(coordlayer):
    (x, y, w, h) = cv2.boundingRect(coordlayer)
    return x, x+w, y, y+h
    
def calcArea(coordlayer):
    if len(coordlayer) > 2:
        coords = [tuple(c) for c in coordlayer]
        sorted_coords = sortCoordsClws(coords)
        X = np.array([item[0] for item in sorted_coords])
        Y = np.array([item[1] for item in sorted_coords])
        correction = X[-1] * Y[0] - Y[-1]* X[0]
        main_area = np.dot(X[:-1], Y[1:]) - np.dot(Y[:-1], X[1:])
        area = 0.5*np.abs(main_area + correction)
        print(area)
        return area
    return 0

def calcDensity(coordlayer, area):
    if len(coordlayer) > 0:
        density = area/len(coordlayer)
        return density
    return 0

#--------------------------------helper funcs-----------------------------------
def getCenter(coords):
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, \
            x, y), coords), [len(coords)] * 2))
    return center

def sortCoordsClws(coords):
    center = getCenter(coords)
    s_coords = sorted(coords, key=lambda coord: (-135 - math.degrees(\
            math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
    return s_coords

def tupleToOppositeVertices(tupleRect):
    (xLeft, yBottom, xRight, yTop) = tupleRect
    vertexMin = (xLeft, yBottom)
    vertexMax = (xRight, yTop)
    return vertexMin, vertexMax

################################################################################

def main():

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True,
        help="-v or --video <path to input video file>")
    args = vars(ap.parse_args())

    #trying out HOG to detect people
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cv2.startWindowThread()

    cap = cv2.VideoCapture(args["video"])

    while cap.isOpened():
        # get data
        success, frame = cap.read()
        frame, coordlayer = getCoords(frame,hog)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # process it
        howManyDancers = calcNumDancers(coordlayer)
        boundingBox = calcBoundingBox(coordlayer)
##        area = calcArea(coordlayer)
##        density = calcDensity(coordlayer,area)

        # display it
        print("There are this many dancers:", howManyDancers)
        showCoords(coordlayer)
        print("Here are coordinates of bounding box: ", boundingBox)

        lowerLeft, upperRight = tupleToOppositeVertices(boundingBox)
        showOutline(frame, lowerLeft, upperRight)
        cv2.imshow('testvideo', frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
