import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False, type=str,
	help="input image", default=None)
ap.add_argument("-v", "--video", required=False, type=str,
	help="input video", default=None)
ap.add_argument("-f", "--frame_ratio", required=False, type=int,
	help="ratio of frames in og vid to frames in altererd vid", default=5)
ap.add_argument("-m", "--mode", required=False, type=str,
	help="type of altering", default='blur_faces', choices=['blur_faces', 'block_eyes'])
ap.add_argument("-t", "--testing", required=False, type=bool,
	help="displays when testing, saves when not", default=False)
args = ap.parse_args()

output_prefix = 'output/'+args.mode+'_'

face_cascade = cv.CascadeClassifier('detectors/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('detectors/haarcascade_eye_tree_eyeglasses.xml')

def blur_faces(img):
    ''' takes image as input, returns same image with faces blurred '''
    # convert to gray scale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # get faces
    faces = face_cascade.detectMultiScale(gray, 1.05, 10)
    for (x, y, h, w) in faces:
        crop = img[y:y+w, x:x+h, :]
        img[y:y+w, x:x+h, :] = cv.blur(crop, (25,25))
    return img

def block_eyes(img):
    ''' Blacks out any eyes found in the image. thought maybe this 
        could do better than the face cascade but I don't think so. '''
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # gete eyes
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=4)
    for (x,y,w,h) in eyes:
        crop = img[y:y+w, x:x+h, :]
        img[y:y+w, x:x+h, :] = np.zeros(crop.shape)

def scrub_video(vid_path):
    ''' alters each frame of video based on mode (face blurred or eyes blocked).
        for time sake, compresses video by frame_ratio (with default of 5, we only
        keep a fifth of the frames). writes altered video to output directory '''
    vid = cv.VideoCapture(vid_path)
    success, image = vid.read()
    size = image.shape[:2][::-1]
    ctr = 0
    fps = vid.get(cv.CAP_PROP_FPS)
    frames = []
    while success:
        ctr += 1
        if ctr % args.frame_ratio == 0:
            if args.mode == 'blur_faces':
                blur_faces(image)
            else: 
                block_eyes(image)
            frames.append(image)
        success, image = vid.read()
    out_path = output_prefix + args.video[:-4] + '.avi' # switches extension to avi
    out = cv.VideoWriter(out_path, cv.VideoWriter_fourcc(*'MJPG'), int(fps/args.frame_ratio), size, True)
    for i in range(len(frames)):
        # writing to a image array
        out.write(frames[i])
    out.release()

def try_all_images():
    ''' checks the result on a testbed of images (for testing purposes)'''
    for root, dirs, files in os.walk('./images'):
        for f in files:
            print(f)
            img = cv.imread('./images/'+f)
            # blur_img = blur_faces(img)
            block_eyes(img)
            # matplotlib.pyplot expects RGB, so we reverse color channels (BGR -> RGB)
            img = img[:,:,::-1]
            plt.imshow(img)
            plt.show()

def main():
    if args.input is not None:
        img = cv.imread(args.input)
        if args.mode == 'blur_faces':
            img = blur_faces(img)
        if args.mode == 'block_eyes':
            block_eyes(img)
        if args.testing:
            img = img[:,:,::-1]
            plt.imshow(img)
            plt.show()
        else:
            img_name = args.input.split('/')[-1]
            cv.imwrite(output_prefix+img_name, img)

    if args.video is not None:
        scrub_video(args.video)
        

main()