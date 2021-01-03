"""
Use CSRT tracker to let user choose object and track that object
returns: list of coordinates indicating the change in position of chosen object
-------------------------------------------------------------------------------
Emma Jin, created 2020.1.3
-------------------------------------------------------------------------------
Tutorials used:
https://www.youtube.com/watch?v=1FJWXOO1SRI
https://www.youtube.com/watch?v=-RtVZsCvXAQ&list=PLS1QulWo1RIa7D1O6skqDQ-JZ1GGHKK-K&index=5
-------------------------------------------------------------------------------
link to download video: https://www.pexels.com/video/black-and-white-video-of-people-853889/
"""

import cv2

cap = cv2.VideoCapture('peoplewalking.mp4')

######################## use CSRT tracker ############################
tracker = cv2.TrackerCSRT_create()
success,frame = cap.read()
box = cv2.selectROI("tracking",frame,False)
tracker.init(frame,box)
coords = []

while cap.isOpened():
    start = cv2.getTickCount()
    success,frame = cap.read()
    success,box = tracker.update(frame)

    if success:
        ###################### draw box #########################
        x,y,w,h = int(box[0]),int(box[1]),int(box[2]),int(box[3])
        coords.append((x,y))
        cv2.rectangle(frame,(x,y),((x+w),(y+h)),(255,0,0),3,1)
        cv2.putText(frame,"tracking",(75,75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    else:
        cv2.putText(frame,"lost",(75,75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    fps = cv2.getTickFrequency()/(cv2.getTickCount()-start)
    cv2.putText(frame,str(int(fps)),(75,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    cv2.imshow('test',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(coords)
