# we can go two approach try and 
# 1) do fast facial landmark with static video and EAR thing  
# 2) or we can go live approach and try EAR ratio in that 

# -------------------- 2nd approach ----------------------------

# i am not going to output any frame in this code
# just i will tell u the number of blinks per minute

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
import copy

import pandas as pd
import numpy
from pandas import DataFrame
from matplotlib import pyplot
from pandas import read_csv
#from pandas import to_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import AdaBoostClassifierfication_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

dataset=pd.read_csv("balanced_preproc_all.csv", index_col="frame")
# Split-out validation dataset
array = dataset.values
X = array[:,:dataset.shape[1]-1].astype(float)
Y = array[:,dataset.shape[1]-1]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,test_size=validation_size, random_state=seed)


# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'accuracy'


# prepare the model
# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = SVC(C=1.7)  #choose our best model and C
model.fit(rescaledX, Y_train)

# estimate accuracy on validation dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

print(roc_auc_score(Y_validation,predictions))


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

def process_ear_lis(ear_lis):
	listear = np.array(ear_lis)
	listear = (listear-np.nanmin(listear))/(np.nanmax(listear)-np.nanmin(listear))
	ear_lis = list(listear)
	# pyplot.rcParams['figure.figsize'] = [24, 24]
	# pyplot.plot(ear_lis)
	# pyplot.savefig('2.png')
	# pyplot.show()
	ear_to_be_fed_into_SVM = []
	for i in range(len(ear_lis) - 6):
		temp = []
		for j in range(7):
			temp.append(ear_lis[i+j])
			# temp.append(ear_lis[i+j])
			# temp.append(ear_lis[i+j])
		ear_to_be_fed_into_SVM.append(np.asarray(temp[:7]))
		# ear_to_be_fed_into_SVM.append(np.asarray(temp[7:14]))
		# ear_to_be_fed_into_SVM.append(np.asarray(temp[14:21]))
	# print(ear_to_be_fed_into_SVM)
	a = np.asarray(ear_to_be_fed_into_SVM, dtype = float)
	# print(type(a))
	# print(a)
	rescaledX = scaler.transform(a)
	# print(rescaledX)
	predictions = model.predict(rescaledX)
	# print(predictions)
	BLINK_LIST = list(predictions)
	for n in range(len(BLINK_LIST)):
		#trovo il primo 1.0
		if BLINK_LIST[n]==1.0:
			i = copy.deepcopy(n)
		#correggi 1.0 isolati: se è un 1.0 singolo (o doppio) diventa 0.0 (o 0.0 0.0)
			if sum(BLINK_LIST[i:i+6])<3.0:
					BLINK_LIST[i]=0.0
			else:
				#correggi 0.0 isolati: se ci sono 0.0 singoli (o doppi) (o tripli) diventano 1.0 (o 1.0 1.0) (o 1.0 1.0 1.0)
				while (sum(BLINK_LIST[i:i+6])>=3.0):
					BLINK_LIST[i+1]=1.0
					BLINK_LIST[i+2]=1.0
					i+=1

	# print(sum(BLINK_LIST))
	#ora costruisco singoli 1.0 corrispondenti al blink
	for n in range(len(BLINK_LIST)):
		#trovo il primo 1.0
		if BLINK_LIST[n]==1.0:
			i = copy.deepcopy(n)
			while (BLINK_LIST[i+1]==1.0):
				BLINK_LIST[i+1]=0.0
				i+=1

	#scala gli 1.0 di 5 frame per posizionarlo alla chiusura circa
	BLINK_LIST=[0.0,0.0,0.0,0.0,0.0]+BLINK_LIST[:len(BLINK_LIST)-5]
	# print(BLINK_LIST)
	print(sum(BLINK_LIST))

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
EYE_AR_THRESH = 0.25 # i changed this from 0.3 to 0.25
EYE_AR_CONSEC_FRAMES = 1 # ------- change i changed this from 2 to 1

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
# vs = FileVideoStream(args["video"]).start()
# fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)


# to process the fast detection, as we told before in the above theory we have to fixed the size of the frame and run the
# face detection and landmarks on that frame and later we scale the output co-ordinates value with the original frame. So
# here we kept the size of the frame is 480.
heightResize = 480

# here we are specifying how many frames it has to skipped in the video file stream so that it will deduct fast face detection. 
a_batch = 2
noofframes_to_process = 1
noofframes_to_skip = 0
temp_batch_counter = 1
temp_to_skip_counter = 1
temp_to_process_counter = 1
# i will process the frames 
# then i will skip it

# loop over frames from the video stream
# frame count that we need
frame_counter = 0
frame_processed_counter = 0
timed_frame_counter = 0
rate_of_blinks = 0 # rate of blinks per minute

ear_lis = []
ear =0

while True:

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

	# we are throwing even frames , as we are going to process only the odd frames
	if (temp_to_process_counter <= noofframes_to_process):
		frame_processed_counter += 1
		if frame is not None:
			frame = imutils.resize(frame, width=450)
		else:
			break
		#frame = imutils.resize(frame, width = 450, height = 450) # ____________________ (1)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# detect faces in the grayscale frame
		rects = detector(gray, 0)

		# loop over the face detections
		for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# extract the left and right eye coordinates, then use the
			# coordinates to compute the eye aspect ratio for both eyes
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)

			# average the eye aspect ratio together for both eyes
			ear = (leftEAR + rightEAR) / 2.0

			# can i delete this it is just visulazing eye points
			# # compute the convex hull for the left and right eye, then
			# # visualize each of the eyes
			# leftEyeHull = cv2.convexHull(leftEye)
			# rightEyeHull = cv2.convexHull(rightEye)
			# cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			# cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

			# check to see if the eye aspect ratio is below the blink
			# threshold, and if so, increment the blink frame counter
			if ear < EYE_AR_THRESH:
				COUNTER += 1
				#print("Counter inc", COUNTER)

			# otherwise, the eye aspect ratio is not below the blink
			# threshold
			else:
				# if the eyes were closed for a sufficient number of
				# then increment the total number of blinks
				if COUNTER >= EYE_AR_CONSEC_FRAMES:
					TOTAL += 1
				# reset the eye frame counter
				COUNTER = 0
			
			# draw the total number of blinks on the frame along with
			# the computed eye aspect ratio for the frame
			#cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			#	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			#cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			#	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		
		temp_to_process_counter += 1
		temp_batch_counter += 1
		# show the frame
		#cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
	elif (temp_to_skip_counter <= noofframes_to_skip):
		temp_to_skip_counter += 1
		temp_batch_counter += 1
	else: 
		temp_batch_counter = 1
		temp_to_process_counter = 1
		temp_to_skip_counter = 1
	
	# increment frame counter 
	frame_counter += 1
	# as we are processing every seconds 15 frames
	# so i calculated that every 10 seconds u need to atleast blink 2 times
	if timed_frame_counter == 300:
		process_ear_lis(ear_lis)
		ear_lis = []
		timed_frame_counter = 0
		# if (TOTAL < 15) :
		# 	print("BLINK BRO !") 
		# 	timed_frame_counter = 0
		# 	TOTAL = 0
	else:
		timed_frame_counter += 1
		ear_lis.append(ear)


print("\n")
print("\nThe total frames in video {}".format(frame_counter))
print("\nThe FPS of the original video is total (frames / duration) of the video")
print("\nThe number of frames processed is {}".format(frame_processed_counter))
print("\nThe total number of blinks is {}".format(TOTAL));
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()