"""
detects dancers from video/camera and analyzes math properties
Emma Jin   created: 2021/1/10
--------------------------------------------------------------------
tutorials used: https://thedatafrog.com/en/articles/human-detection-video/
https://stackoverflow.com/questions/51074984/sorting-according-to-clockwise-point-coordinates/51075469
https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
https://stackoverflow.com/questions/14452145/how-to-measure-time-taken-between-lines-of-code-in-python
https://zetcode.com/python/argparse/
"""
import cv2
import numpy as np
from functools import reduce
import operator
import math
import time
import argparse as ap

################################# CONSTANTS ####################################
WIDTH = 1080
HEIGHT = 720

POS_HIGH = HEIGHT // 15
POS_MID = POS_HIGH + HEIGHT // 3
POS_LOW = POS_MID + HEIGHT // 3
POS_LEFT = WIDTH // 20
POS_CENTER = POS_LEFT + WIDTH // 3
POS_RIGHT = POS_CENTER + WIDTH // 3

BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)
BLACK = (0,0,0)
WHITE = (255,255,255)
CYAN = (255,255,0)
YELLOW = (0,255,255)
############################## COMMAND LINE ARGUMENTS ##########################
parser = ap.ArgumentParser()
parser.add_argument('-v', '--filename', default='peoplewalking.mp4', \
    help='specify the path and filename of input video file')
args = parser.parse_args()
filename = args.filename
print(filename)

############################## PRESENTATION TIER ###############################
def showRects(frame, rects, color):
    """
    display rectangles in frame given the rectangles' vertices
    """
    for (p1,p2,p3,p4) in rects:
        cv2.rectangle(frame, (p1,p2), (p3,p4), color, 2)
    return frame

def showCoords(frame, coord):
    """
    display a point in frame given its coordinate
    """
    coord = tuple(int(item) for item in coord)
    cv2.circle(frame, coord, radius=10, color=RED, thickness=2)
    return frame

def showFeature(frame, s, data, pos_w = POS_LEFT, pos_h = POS_HIGH, pos_num = None):
    """
    display a certain feature in frame given the feature's name, data, and position
    (pos_num is a tuple in the form of (x,y))
    """
    if pos_num != None:
        pos = pos_num
    else:
        pos = (pos_w, pos_h)
    cv2.putText(frame,"%s:%.2f"%(s,data), pos, cv2.FONT_HERSHEY_SIMPLEX,\
                1, BLUE, 2, cv2.LINE_AA)
    return frame

def showPolygon(frame, coords):
    """
    display the polygon connecting a set of coordinates
    """
    coords = sortCoordsClws(coords)
    coords = np.asarray(coords)
    coords = coords.reshape((-1,1,2))
    frame = cv2.polylines(frame,[coords],True,YELLOW)
    return frame

def showContours():
    pass

################################## DATA TIER ###################################
def getCoords(frame, hog):
    """
    get the coords of detected dancers in a certain frame
    returns: coords, a numpy array of two-element-lists, each representing a coord
    """
    frame = cv2.resize(frame, (1080, 720))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = getRects(frame,hog)
    frame = showRects(frame, rects, GREEN)
    coords = np.array([[int((x1+x2)/2), int((y1+y2)/2)] for (x1,y1,x2,y2) in rects])
    return frame, coords

def getRects(frame, hog):
    """
    get the coords of the lower-left and upper-right vertices for bounding
    rectangles around dancers
    returns: rects, a numpy array of four-element-lists, each a vertex
    """
    rects = getRects_init(frame, hog)
    rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
    return rects

def getRects_init(frame, hog):
    rects, _ = hog.detectMultiScale(frame, winStride=(4,8)) #play with this and the resize?
    return rects

def checkRects(rects):
    """
    delete any rectangles that are surrounding multiple other rectangles
    returns: array of the remaining smaller rectangles
    """
    #for rect in rects:
        #if rect[2] > WIDTH//2 or rect[3] > HEIGHT//2
    pass

def getCenterContour(coords):
    pass

############################# APPLICATION TIER #################################
def getCenter(coords):
    """
    get the center of a set of coordinates
    returns: the coords of the calculated center (type tuple)
    """
    s = coords.shape
    sums = np.sum(coords, axis=0)
    cx = sums[0]/s[0]
    cy = sums[1]/s[0]
    return (cx,cy)

def getBoundingRect(coordlayer):
    """
    get bounding rectangle using cv2.contours given a list of coordinates
    """
    x,y,w,h = cv2.boundingRect(coordlayer)
    rect = (x,y,x+w,y+h)
    return [rect]

def calcArea(coordlayer):
    """
    calculate the area of the polygon connected by a set of coordinates
    """
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

def calcAreaContour(coordlayer):
    pass

def calcDensity(coordlayer, area):
    """
    calculate the density of dancers in the space bounded by the polygon
    """
    density = area/len(coordlayer)
    return density

def calcAveVelocity(center1, center2, delta_t):
    """
    calculate the average velocity of the group of dancers
    """
    dist = math.hypot(center2[0]-center1[0], center2[1]-center1[1])
    v = dist/delta_t
    return v

#--------------------------------helper funcs-----------------------------------
def getRotationCenter(coords):
    """
    helper function to get rotation center for a set of coords
    """
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, \
            x, y), coords), [len(coords)] * 2))
    return center

def sortCoordsClws(coords):
    """
    helper function to sort a set of coordinates in clockwise order
    """
    center = getRotationCenter(coords)
    s_coords = sorted(coords, key=lambda coord: (-135 - math.degrees(\
            math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
    return s_coords

################################################################################

def main():
    #trying out HOG to detect people
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cv2.startWindowThread()
    cap = cv2.VideoCapture(filename)
    #cap = cv2.VideoCapture(0) #number could vary depending on camera
    oldcenter = None #
    framecount = 1
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            frame, coordlayer = getCoords(frame,hog)
            if coordlayer.size != 0:
                boundingrect = getBoundingRect(coordlayer)
                frame = showRects(frame, boundingrect, CYAN)
                center = getCenter(coordlayer)
                end = time.time() #
                frame = showCoords(frame,center)
                area = calcArea(coordlayer)
                density = calcDensity(coordlayer,area)
                frame = showFeature(frame,"density",density)
                frame = showPolygon(frame,coordlayer)

                if oldcenter != None:
                    aveVelocity = calcAveVelocity(oldcenter,center,end-start)
                    frame = showFeature(frame,"ave. velocity",aveVelocity,\
                    pos_num=(WIDTH//20,HEIGHT//8))

                start = time.time() #
                oldcenter = center #

            cv2.imshow('testvideo', frame)
        else:
            print("Failed to read frame %d" %framecount)
            # if keeps detecting continuous failed frames, break since we reached
            # the end of the video?
            if framecount == old_framecount + 1:
                break
        old_framecount = framecount
        framecount += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
