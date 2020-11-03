import cv2
import math
import numpy as np
from tensorflow import keras
from typing import List, NewType
from matplotlib import pyplot as plt


ndarray = NewType("numpy ndarray", np.ndarray)

def get_face_hue()-> int:
	cap = cv2.VideoCapture(0)
    # create a copy of the image to prevent any changes to the original one.
	while(True):
		ret, frame = cap.read()
		image_copy = frame.copy()

		gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

		cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

		face_rect = cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

		for (x, y, w, h) in face_rect:

			w_offset, h_offset = w//10, h//10

			face = frame[y+h_offset:y+h-h_offset, x+w_offset:x+w-w_offset]

			return int(cv2.cvtColor(face, cv2.COLOR_BGR2HSV)[:,:,0].mean())

def frame_by_frame()-> ndarray:
	cap = cv2.VideoCapture(0)
	mfh = get_face_hue()
	hands = []
	while(True):
	    # Capture frame-by-frame
	    ret, frame = cap.read()

	    hsvim = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	    lower = np.array([mfh - (mfh//2), 50, 50], dtype = "uint8")
	    upper = np.array([mfh + (mfh//2) , 255, 255], dtype = "uint8")
	    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
	    blurred = cv2.blur(skinRegionHSV, (2,2))
	    ret,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)

	    h, w, _ = frame.shape
	    canvas = thresh.copy()

	    canvas[:h//5,  :] = 0
	    canvas[-h//5:, :] = 0
	    canvas[:,  int(w*0.4):int(w*0.6)] = 0
	    canvas[:, w//2] = 255

	    image, contours, hierarchy = cv2.findContours(canvas, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	    if contours:
		    largest_contours = sorted(contours, key=lambda x: cv2.contourArea(x))[-3:] # top 3
		    for lc in largest_contours:
			    if (w*h//16 < cv2.contourArea(lc) < w*h//8) and (1/2 < h/w < 2):
			    	#x,y,w,h = cv2.boundingRect(c)
			    	hull = cv2.convexHull(lc, returnPoints=False)
			    	defects = cv2.convexityDefects(lc, hull)
			    	for i in range(np.size(defects, 0)):
				    	spt = tuple(lc[defects[i,0,0]].flatten())
				    	fpt = tuple(lc[defects[i,0,1]].flatten())
				    	ept = tuple(lc[defects[i,0,2]].flatten())


				    	cv2.line(frame, spt, ept, (0,255,0), 2)
				    	cv2.line(frame, fpt, ept, (0,0,255), 2)
		
			    	#cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 5)
			    	#h_off, w_off = h//10, w//10
			    	#grey_hand = frame[y-h_off:y+h+h_off, x-w_off:x+w+w_off, 2]
			    	#new_hand = cv2.resize(frame[y-h_off:y+h+h_off, x-w_off:x+w+w_off, 2], (128,128))
			    	#if i == 0: hands = new_hand
			    	#else: hands = np.dstack((hands, new_hand))
			    	#hands.append(new_hand)
			    	#i += 1
	    
	    # cv2.imshow("lines", frame)
	    cv2.imshow("img", frame)
	    if (cv2.waitKey(1) & 0xFF == ord('q')) or (len(hands) > 50):
	        break
	    print(len(hands))

	cap.release()
	cv2.destroyAllWindows()
	return np.asarray(hands)

if __name__ == '__main__':

	#x = cv2.imread('../train/1040e0bf-a518-49e3-a720-78ca4fcdedf6_2R.png')
	#cv2.imshow("hand", x[:,:,0])
	#cv2.waitKey(0)
	#print(x.shape)

	#exit()

	#model = keras.models.load_model('turkey_hunter')
	
	hands = frame_by_frame()
	#hands = hands[..., np.newaxis] # keras needs a channel dim

	#predictions = model.predict_classes(hands)

	#for i in range(np.size(hands,2)):
	#	h = hands[:,:,i]
	#	cv2.imshow("hand",h)
	#	cv2.waitKey(0)
	#cv2.destroyAllWindows()  
