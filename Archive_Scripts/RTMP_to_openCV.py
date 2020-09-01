import numpy as np
import cv2

# Open a sample video available in sample-videos
myadder="srt://192.168.1.202:1935?streamid=output/live/atl"
myadder2="rtmp://192.168.1.202/live/test"
vcap = cv2.VideoCapture(myadder2)

fgbg4 = cv2.createBackgroundSubtractorKNN(history = 25, dist2Threshold= 300.0 ,detectShadows=True); 


while(True):
    # Capture frame-by-frame
    ret, frame = vcap.read()

    if frame is not None:
        # turn to greyscale:
        frame2 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # apply threshold. all pixels with a level larger than 80 are shown in white. the others are shown in black:
        ret,frame2 = cv2.threshold(frame2,80,255,cv2.THRESH_BINARY)
        

        fgmask4 = fgbg4.apply(frame2)
        roughOutput = cv2.bitwise_and(frame, frame, mask=fgmask4)
        cv2.imshow('frame',frame2)
        cv2.imshow('frame2',frame)

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