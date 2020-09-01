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
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import pickle

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-tiny-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', 'rtmp://192.168.1.202/live/test', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'avi', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.35, 'iou threshold')
flags.DEFINE_float('score', 0.30, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')


def main(_argv):
    tv=1
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video
    FPS = 1/60
    FPS_MS=int(FPS*1000)

    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    vcap = cv2.VideoCapture(video_path)
    vcap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    while(True):
        ret, frame = vcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        tv = printPickle()

        if tv['channel'] == 1:
            result = objDetect(frame, input_size, FLAGS.iou, FLAGS.score, infer)
        elif tv['channel'] == 2:
            result = objDetect(frame, input_size, FLAGS.iou, FLAGS.score, infer)
            result = blame_vision(result)
        elif tv['channel'] == 3:
            result = cartoonize(frame,2,3)

        if frame is not None:
            result = np.asarray(result)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("result",result)

        if cv2.waitKey(FPS_MS) & 0xFF == ord('q'):
            break
    vcap.release()
    cv2.destroyAllWindows()
    print("Video stop")

def blame_vision(img):
    blur = cv2.GaussianBlur(img,(5,5),0)
    filtered = img - blur
    return filtered

def cartoonize(img,samp, bi):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)

    img = cv2.bilateralFilter(img, 9, 300, 300)

    cartoon = cv2.bitwise_and(img, img, mask=edges)
    return cartoon



def objDetect(fr, sz, iou, scor, infr):
    image = Image.fromarray(fr)
    frame_size = fr.shape[:2]
    image_data = cv2.resize(fr, (sz, sz))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    start_time = time.time()
    batch_data = tf.constant(image_data)
    pred_bbox = infr(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou,
        score_threshold=scor
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    image = utils.draw_bbox(fr, pred_bbox)
    image = utils.draw_thing(fr, pred_bbox)
    #fps = 1.0 / (time.time() - start_time)
    #print("FPS: %.2f" % fps)
    
    return image

def printPickle():
    with open('..\Stream\effectChannel.pickle','rb') as chan:
        chanNum = pickle.load(chan)
        return chanNum



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
    