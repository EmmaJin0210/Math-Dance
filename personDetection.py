"""
returns coords of dancers
Emma Jin   created: 2021/1/10
--------------------------------------------------------------------
tutorial used: https://thedatafrog.com/en/articles/human-detection-video/
https://stackoverflow.com/questions/51074984/sorting-according-to-clockwise-point-coordinates/51075469
https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates

"""

import cv2
import numpy as np
from functools import reduce
import operator
import math


############################## PRESENTATION TIER ###############################
def showRects(frame, rects):
    for (p1,p2,p3,p4) in rects:
        cv2.rectangle(frame, (p1,p2), (p3,p4), (0,255,0), 2)
    return frame

def showCoords(frame, coord):
    coord = tuple(int(item) for item in coord)
    cv2.circle(frame, coord, radius=10, color=(0, 0, 255), thickness=2)
    return frame

def showDensity(frame, density):
    cv2.putText(frame,"density:%.2f"%(density), (50,50),cv2.FONT_HERSHEY_SIMPLEX,\
                1, (255,0,0), 2, cv2.LINE_AA)
    return frame

def showCentroid(frame, coords):
    pass

################################## DATA TIER ###################################
def getCoords(frame, hog):
    frame = cv2.resize(frame, (720, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = getRects(frame,hog)
    frame = showRects(frame, rects)
    coords = np.array([[int(x+w/2), int(y+h/2)] for (x,y,w,h) in rects])
    return frame, coords

def getRects(frame, hog):
    rects, _ = hog.detectMultiScale(frame, winStride=(8,8))
    rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
    return rects
############################# APPLICATION TIER #################################
def calcArea(coordlayer):
    if len(coordlayer) > 2:
        coords = [tuple(c) for c in coordlayer]
        sorted_coords = sortCoordsClws(coords)
        X = np.array([item[0] for item in sorted_coords])
        Y = np.array([item[1] for item in sorted_coords])
        correction = X[-1] * Y[0] - Y[-1]* X[0]
        main_area = np.dot(X[:-1], Y[1:]) - np.dot(Y[:-1], X[1:])
        area = 0.5*np.abs(main_area + correction)
        return area
    return 0

def calcAreaBox(coordlayer):
    pass

def calcDensity(coordlayer, area):
    if len(coordlayer) > 0:
        density = area/len(coordlayer)
        return density
    return 0

def calcDensityBox(coordlayer, area):
    pass
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
################################################################################

def main():
    #trying out HOG to detect people
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cv2.startWindowThread()
    cap = cv2.VideoCapture("peoplewalking.mp4")

    while cap.isOpened():
        success, frame = cap.read()
        frame, coordlayer = getCoords(frame,hog)
        center = getCenter(coordlayer)
        frame = showCoords(frame,center)
        area = calcArea(coordlayer)
        density = calcDensity(coordlayer,area)
        frame = showDensity(frame, density)
        cv2.imshow('testvideo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
