"""
returns coords of dancers
Emma Jin   created: 2021/1/10
--------------------------------------------------------------------
tutorial used: https://thedatafrog.com/en/articles/human-detection-video/
"""

import cv2
import numpy as np


############################## PRESENTATION TIER ###############################

################################## DATA TIER ###################################
def getCoords(frame, hog):
    frame = cv2.resize(frame, (720, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects, _ = hog.detectMultiScale(frame, winStride=(6,6))
    rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
    coords = np.array([[int(x+w/2), int(y+h/2)] for (x,y,w,h) in rects])
    for (p1,p2,p3,p4) in rects:
        cv2.rectangle(frame, (p1,p2), (p3,p4), (0,255,0), 2)
    return frame, coords

############################# APPLICATION TIER #################################

################################################################################

def main():
    #trying out HOG to detect people
    coords = []  #data
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cv2.startWindowThread()
    cap = cv2.VideoCapture("peoplewalking.mp4")

    while cap.isOpened():
        success, frame = cap.read()
        frame, coordlayer = getCoords(frame,hog)
        coords.append(coordlayer)
        cv2.imshow('testvideo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(coords)

if __name__ == "__main__":
    main()
