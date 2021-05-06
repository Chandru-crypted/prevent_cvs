from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import cv2
import time
# start the video stream thread
print("[INFO] starting video stream thread...")
# vs = FileVideoStream(args["video"]).start()
# fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)

frame_counter = 0 

fourcc = cv2.VideoWriter_fourcc('F', 'F', 'V', '1')
#out = cv2.VideoWriter(filename = 'output.mkv', fourcc = fourcc, fps = 30.0,frameSize = (480,480))

#writer = cv2.VideoWriter("output1.mkv", -1, 30.0,
#		(450, 450), True)
writer = cv2.VideoWriter("output1.avi", fourcc, 30.0,(640, 480) , True)
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
	#cv2.putText(frame, "Frame no: {}".format(frame_counter), (10, 30),
	#		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	writer.write(frame)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	if frame_counter == 300:
		break
	frame_counter += 1 


writer.release()
cv2.destroyAllWindows()
vs.stop()