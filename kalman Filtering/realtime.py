import cv2
from frame_processing import frame_process

vid = cv2.VideoCapture(0)

while(True):
	ret, frame = vid.read()

	cv2.imshow('frame', frame_process(frame))
    
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

vid.release()
cv2.destroyAllWindows()
