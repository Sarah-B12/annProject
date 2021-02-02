import numpy as np
import cv2 as cv
cap = cv.VideoCapture("a.avi")
ret, frame = cap.read()
if ret is False:
    print("Cannot read video stream")
    exit()
myvideo=cv.VideoWriter("aa.avi", cv.VideoWriter_fourcc('M','J','P','G'), 30, (int(frame.shape[1]),int(frame.shape[0])))
fgbg = cv.createBackgroundSubtractorMOG2(100, 100, True)
while(1):
    ret, frame = cap.read()
    if ret is False:
        print("Cannot read video stream")
        exit()
    fgmask = fgbg.apply(frame)
    cv.imshow('frame',fgmask)
    fg = cv.copyTo(frame,fgmask)
    myvideo.write(fg)
    cv.imshow('Foreground',fg)
    cv.imshow('Background',cv.copyTo(frame,cv.bitwise_not(fgmask)))
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
myvideo.release()
cv.destroyAllWindows()