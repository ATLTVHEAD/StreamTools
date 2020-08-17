import os
import tempfile
import atexit
import subprocess
import cv2

FFMPEG_BIN = '/usr/bin/ffmpeg'

def run_ffmpeg(fifo_path):
    ffmpg_cmd = [
        FFMPEG_BIN,
        '-i', '/dev/video0',
        '-r', '1',
        '-pix_fmt', 'bgr24',        # opencv requires bgr24 pixel format.
        '-vcodec', 'rawvideo',
        '-an','-sn',                # disable audio processing
        '-f', 'yuv',
        fifo_path 
    ]
    return subprocess.Popen(ffmpg_cmd)


def run_cv_window(fifo_path):
    cap = cv2.VideoCapture(fifo_path)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def run():
    fifo_path = tempfile.mktemp(suffix='img_pipe')
    os.mkfifo(fifo_path)
    atexit.register(os.unlink, fifo_path)
    ffmpeg_process = run_ffmpeg(fifo_path)
    run_cv_window(fifo_path)

if __name__ == '__main__':
    run()