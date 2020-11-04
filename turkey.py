import cv2
import time
import math
import numpy as np
from typing import List, NewType, Set, Tuple
from matplotlib import pyplot as plt


ndarray = NewType("numpy ndarray", np.ndarray)
VideoIO = NewType("cv2 webcam", cv2.VideoCapture)

def compute_centroid(points: List[ndarray])->Tuple[int, int]:
	points = np.asarray(points)
	c = points.mean(axis=0)
	return tuple(c.astype(int).flatten())


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


def initialize(cap: VideoIO, init_time: int=5)-> Tuple[int, Set[Tuple[int, int]]]:
	mfh = get_face_hue()
	bad_points = set()

	t_end = time.time() + init_time
	while time.time() < t_end:
		new_centroid = find_hands(cap, mfh, bad_points, initializing=True)
		if new_centroid:
			bad_points.add(new_centroid)
			print("GOT ONE")

	return mfh, bad_points

dst = lambda p1, p2: np.sqrt(np.sum(np.square(p2-p1)))

def find_hands(cap: VideoIO, mfh: int, bad_points: Set[Tuple[int, int]]= None,
	              					   		 initializing: bool=False)-> None:
	hands = []
	centroid = None
	min_frac, max_frac = 32, 8 # experimentally determined

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

	    thresh[:int(h*0.25),  :]  = 0
	    thresh[int(h*0.75):, :]  = 0
	    thresh[:, int(w*0.4):int(w*0.6)] = 0

	    top_n = -3 if not initializing else 0

	    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	    if contours:
		    largest_contours = sorted(contours, key=lambda x: cv2.contourArea(x))[top_n:]

		    for lc in largest_contours:

			    if (w*h//min_frac < cv2.contourArea(lc) < w*h//max_frac) and (1/2 < h/w < 2):

			    	hull = cv2.convexHull(lc, returnPoints=False)
			    	defects = cv2.convexityDefects(lc, hull)
			    	
			    	pts = []
			    	for i in range(np.size(defects, 0)):
				    	spt = lc[defects[i,0,0]].flatten()
				    	fpt = lc[defects[i,0,1]].flatten()
				    	ept = lc[defects[i,0,2]].flatten()

				    	if dst(spt, ept) > 300 or dst(fpt, ept) > 300: continue # too long

				    	pts.extend([spt, fpt, ept])

			    	centroid = compute_centroid(pts)
			    	if bad_points and \
			    	   sum([dst(centroid, np.array(p))<= 10 \
			    	   for p in bad_points]) >= 3: continue


			    	cv2.circle(frame, centroid, radius=5, color=(0, 255, 0), thickness=-1)
	    
	    cv2.imshow("img", frame)
	    if (cv2.waitKey(1) & 0xFF == ord('q')) or (len(hands) > 50): break
	    if initializing: return centroid

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':

	cap = cv2.VideoCapture(0)
	
	print("\tInitializing...")
	mfh, bad_points = initialize(cap, 20)
	print("\tWe're live!")
	find_hands(cap, mfh, bad_points)

