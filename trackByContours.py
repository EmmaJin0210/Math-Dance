"""
Detects people moving in video through contour curves
prints: list of coordinates of all detected moving objects
"""
#-------------------------------------------------------------------------------
#Emma Jin, created 2020.1.3
#-------------------------------------------------------------------------------
#Tutorials used:
#https://www.youtube.com/watch?v=-RtVZsCvXAQ&list=PLS1QulWo1RIa7D1O6skqDQ-JZ1GGHKK-K&index=5
#https://www.youtube.com/watch?v=MkcUgPhOlP8&list=PLS1QulWo1RIa7D1O6skqDQ-JZ1GGHKK-K&index=28
#-------------------------------------------------------------------------------
#link to download video: https://www.pexels.com/video/black-and-white-video-of-people-853889/


import cv2
import numpy as np

size = int(input("What is the approximate size of people in the video? "))
# gets approximate size of people in video to avoid rectangles being too small
coords = []

cap = cv2.VideoCapture('peoplewalking.mp4')

success, frame1 = cap.read()
success, frame2 = cap.read()

while cap.isOpened():
    if success:
        ########################GET CONTOURS########################
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _ , thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #######################DRAW RECTANGLES######################
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < size:
                continue
            coords.append((x,y))
            cv2.rectangle(frame1, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame1, "People Detected", (10,20),
                        1, cv2.FONT_HERSHEY_SIMPLEX, (0,0,255), 3)

        #############################################################
        cv2.imshow("testvid", frame1)
        frame1 = frame2
        success, frame2 = cap.read()

        if cv2.waitKey(40) == 27:
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()
print(coords)
