"""
Main file for implementing motion detection and clip saving
"""


import cv2, numpy, sys

from datetime import datetime

suffix = sys.argv[1]

CAMERA_INDEX = 0

backSub = cv2.createBackgroundSubtractorKNN()

video = cv2.VideoCapture(CAMERA_INDEX)
if not video.isOpened():
    print('Unable to open camera')
    exit(0)
video.set(3,1280)
video.set(4,720)

writer= cv2.VideoWriter('recording_' + str(suffix) + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

rgb = (51,255,51)
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 0.8

while True:
    
    check, frame = video.read()
    if frame is None:
        break
    
    fgMask = backSub.apply(frame)
    
    num_white = numpy.count_nonzero(fgMask==255)
    
    cv2.putText(frame, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), (20, 30),
               font, scale, rgb)
    cv2.putText(frame, 'Pixels: ' + str(num_white), (20, 60),
               font, scale, rgb)
    
    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)
    
    
    if num_white > 200: 
        writer.write(frame)
        
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
    
  
video.release()
writer.release()
cv2.destroyAllWindows()