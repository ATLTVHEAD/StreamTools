import numpy as np
import cv2
from matplotlib import pyplot as plt


# Open a sample video available in sample-videos
myadder="srt://192.168.1.202:1935?streamid=output/live/atl"
myadder2="rtmp://192.168.1.202/live/test"
cap = cv2.VideoCapture(myadder2)


#img = cv2.imread('messi5.jpg')






#plt.imshow(img),plt.colorbar(),plt.show()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #print cap.isOpened(), ret
    if frame is not None:
        mask = np.zeros(frame.shape[:2],np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = (25,25,1280,720)
        cv2.grabCut(frame,mask,rect,bgdModel,fgdModel,20,cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        frame = frame*mask2[:,:,np.newaxis]

        #plt.imshow(frame),plt.colorbar(),plt.show()
        cv2.imshow('frame',frame)
        cv2.waitKey(1)
        # Press q to close the video windows before it ends if you want
        if cv2.waitKey(22) & 0xFF == ord('q'):
            break
    else:
        print("nothing yet")
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
print("Video stop")