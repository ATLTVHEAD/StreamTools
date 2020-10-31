import cv2
import argparse
import os

#Arguments to pass in input video file + location, the output file location, the output filename, the fps which effects the number of pngs generated
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Convert Video to PNG Sequence')
    parser.add_argument('--video',
                        default='none', type=str,
                        help='location of your video file "/videos/video.mp4"')
    parser.add_argument('--output_file_location', default='png_sequence', type=str,
                        help='Output file location')
    parser.add_argument('--fps', default=23.98, type=float,
                        help='FPS of video')
    parser.add_argument('--output_filename', default='png_sequence', type=str,
                        help='Output file name')
    parser.set_defaults(video='', output_file_location='png_sequence', resume=24, output_filename='png_seqence')

    global args
    args = parser.parse_args(argv)

#Takes a specific frameout of the video at the msec timing and saves it as a png
def getFrame(msec, dir , video):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, msec)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(dir + '/' + video + '_' + str(count) + '.png', image)
    return hasFrames

#checks for and makes a directory for the photos
def make_dir(dire):
    if not os.path.exists(dire):
        os.makedirs(dire)


if __name__ == '__main__':
    parse_args()
    sec = 0
    frameRate = int(1000/args.fps)
    count=1
    make_dir(args.output_file_location)
    vidcap = cv2.VideoCapture(args.video)
    success = getFrame(sec, args.output_file_location, args.output_filename)
    while success:
        count = count + 1
        print(count)
        sec = sec + frameRate
        success = getFrame(sec, args.output_file_location, args.output_filename)