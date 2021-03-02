# we can go two approach try and 
# 1) do fast facial landmark with static video and EAR thing  
# 2) or we can go live approach and try EAR ratio in that 

# -------------------- 1st approach ----------------------------


# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


def eye_aspect_ratio(eye):
	# eye contains the coordinate of both left and right eye points
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return (ear)


# Usage
# python testing_fast_facial_landmark.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# for argument that we can pass in command prompt
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3 

# Since we are skipping frames in fast processing should we change this const that is above

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = FileVideoStream(args["video"]).start()
fileStream = True
# vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
# fileStream = False
time.sleep(1.0)


# to process the fast detection, as we told before in the above theory we have to fixed the size of the frame and run the
# face detection and landmarks on that frame and later we scale the output co-ordinates value with the original frame. So
# here we kept the size of the frame is 480.
heightResize = 480

# here we are specifying how many frames it has to skipped in the video file stream so that it will deduct fast face detection. 
framesSkipping = 2

# loop over frames from the video stream
while True:
	# frame count that we need
	count = 0


	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break

	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()

	#frame = imutils.resize(frame, width=450)
	# this actually keeps the aspect ratio intact 

	# https://answers.opencv.org/question/208857/what-is-the-difference-between-cv2resize-and-imutilsresize/ 
	# and also in the interpolation the fast facial landmark has given cv2.INTER_LINEAR 
	# but as you check the website we can see that the defualt in imutils is cv2.INTER_AREA

	frame = imutils.resize(frame, width = 450, height = 450) # ____________________ (1)


	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# increment frame counter 
	count += 1 