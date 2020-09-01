import numpy as np 
import cv2

def main():
    vcap = cv2.VideoCapture('rtmp://192.168.1.202/live/test')
    while True:
        ret, frame = vcap.read()
        cartoon = cartoonize(frame,2,3)
        if frame is not None:
            cv2.namedWindow("cartoon", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("cartoon",cartoon)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vcap.release()
    cv2.destroyAllWindows()
    print("Video stop")


def cartoonize(img,samp, bi):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)

    for _ in range(samp):
        img = cv2.pyrDown(img)
    for _ in range(bi):
        img = cv2.bilateralFilter(img, 9, 9, 7)
    for _ in range(samp):
        img= cv2.pyrUp(img)

    cartoon = cv2.bitwise_and(img, img, mask=edges)
    return cartoon


if __name__=='__main__':
    main()