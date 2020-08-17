import numpy as np
import cv2

# Open a sample video available in sample-videos
myadder="srt://192.168.1.202:1935?streamid=output/live/atl"
myadder2="rtmp://192.168.1.202/live/test"
vcap = cv2.VideoCapture(myadder2)
#if not vcap.isOpened():
#    print "File Cannot be Opened"



# creating object 
#fgbg1 = cv2.bgsegm.createBackgroundSubtractorMOG(history=10); 
#fgbg2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold = 16, detectShadows=False); 
#fgbg3 = cv2.bgsegm.createBackgroundSubtractorGMG(); 
fgbg4 = cv2.createBackgroundSubtractorKNN(history = 25, dist2Threshold= 300.0 ,detectShadows=True); 


while(True):
    # Capture frame-by-frame
    ret, frame = vcap.read()
    #print cap.isOpened(), ret
    if frame is not None:
        # turn to greyscale:
        frame2 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # apply threshold. all pixels with a level larger than 80 are shown in white. the others are shown in black:
        ret,frame2 = cv2.threshold(frame2,80,255,cv2.THRESH_BINARY)
        
        # Display the resulting frame
        #fgmask1 = fgbg1.apply(frame)
        #fgmask2 = fgbg2.apply(frame)
        #fgmask3 = fgbg3.apply(frame)
        fgmask4 = fgbg4.apply(frame2)
        roughOutput = cv2.bitwise_and(frame, frame, mask=fgmask4)
        cv2.imshow('frame',frame2)
        cv2.imshow('frame2',frame)
        #cv2.imshow('MOG', roughOutput)
        #cv2.imshow('MOG2', fgmask2)
        #cv2.imshow('GMG', fgmask3)

        # Press q to close the video windows before it ends if you want
        if cv2.waitKey(22) & 0xFF == ord('q'):
            break
    else:
        print("nothing yet")
        break

# When everything done, release the capture
vcap.release()
cv2.destroyAllWindows()
print("Video stop")