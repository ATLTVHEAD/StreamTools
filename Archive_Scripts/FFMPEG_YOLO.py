import time 
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np 
import subprocess as sp
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def main(_argv): 
	IMG_W = 1280
	IMG_H = 720

	saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-tiny-416', tags=[tag_constants.SERVING])
	infer = saved_model_loaded.signatures['serving_default']

	FFMPEG_BIN = "C:/FFmpeg/bin/ffmpeg"
	ffmpeg_cmd = [ FFMPEG_BIN,
				'-i', 'srt://192.168.1.106:1935?streamid=output/live/atl',
				'-r', '24',					# FPS
				'-pix_fmt', 'bgr24',      	# opencv requires bgr24 pixel format.
				'-vcodec', 'rawvideo',
				'-an','-sn',              	# disable audio processing
				'-f', 'image2pipe', '-']    
	pipe = sp.Popen(ffmpeg_cmd, stdout = sp.PIPE, bufsize=10**8)
	first =True
	while True:
		raw_image = pipe.stdout.read(IMG_W*IMG_H*3)
		image =  np.frombuffer(raw_image, dtype='uint8')		# convert read bytes to np
		og_image = image.reshape((IMG_H,IMG_W,3))
		#og_image = cv2.cvtColor(og_image, cv2.COLOR_BGR2RGB)
		image_data = cv2.resize(og_image, (416, 416))
		image_data = image_data / 255.
		images_data = [image_data]
		images_data = np.asarray(images_data).astype(np.float32)
		start_time = time.time()

		batch_data = tf.constant(images_data)
		pred_bbox = infer(batch_data)
		for key, value in pred_bbox.items():
			boxes = value[:, :, 0:4]
			pred_conf = value[:, :, 4:]
		
		boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
			boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
			scores=tf.reshape(
				pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
			max_output_size_per_class=50,
			max_total_size=50,
			iou_threshold=0.45,
			score_threshold=0.25
		)

		pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
		image = utils.draw_bbox(og_image, pred_bbox)
		fps = 1.0 / (time.time() - start_time)
		print("FPS: %.2f" % fps)
		cv2.imshow('Video', image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
			
		pipe.stdout.flush()

	cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass