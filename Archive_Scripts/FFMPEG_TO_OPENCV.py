import time 
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np 
import subprocess as sp


def main(_argv): 
	IMG_W = 1280
	IMG_H = 720

	FFMPEG_BIN = "C:/FFmpeg/bin/ffmpeg"
	ffmpeg_cmd = [ FFMPEG_BIN,
				'-i', 'srt://192.168.1.106:1935?streamid=output/live/atl',
				'-r', '24',					# FPS
				'-pix_fmt', 'bgr24',      	# opencv requires bgr24 pixel format.
				'-vcodec', 'rawvideo',
				'-an','-sn',              	# disable audio processing
				'-f', 'image2pipe', '-']    
	pipe = sp.Popen(ffmpeg_cmd, stdout = sp.PIPE, bufsize=10**8)
	fgbg4 = cv2.createBackgroundSubtractorKNN(history = 200, dist2Threshold= 300.0 ,detectShadows=True); 

	while True:
		raw_image = pipe.stdout.read(IMG_W*IMG_H*3)
		image =  np.frombuffer(raw_image, dtype='uint8')		# convert read bytes to np
		image = image.reshape((IMG_H,IMG_W,3))
		#frame2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		#ret,frame3 = cv2.threshold(frame2,80,255,cv2.THRESH_BINARY)
		#fgmask4 = fgbg4.apply(frame3)
		#roughOutput = cv2.bitwise_and(image, image, mask=fgmask4)

		cv2.imshow('Video', image)
		#cv2.imshow('Video2', frame2)
		#cv2.imshow('Video3', frame3)
		#cv2.imshow('Video4', roughOutput)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
			
		pipe.stdout.flush()

	cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass