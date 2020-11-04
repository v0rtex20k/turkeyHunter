import cv2
import time
import math
import numpy as np
from typing import List, NewType, Set, Tuple
from matplotlib import pyplot as plt

Point   = NewType("xy coordinate", np.ndarray)
ndarray = NewType("numpy ndarray", np.ndarray)
VideoIO = NewType("cv2 webcam stream", cv2.VideoCapture)

def compute_centroid(*points_lists: List[List[Point]])-> Point:
	points = np.asarray(list(zip(*points_lists))).reshape(-1,2)
	return points.mean(axis=0).astype(int).flatten()

def midpt (p1, p2) -> float:
	mpt = 0.5*(p1+p2)
	return mpt.astype(int)

def pdst (p1, p2) -> float:
	return np.sqrt(np.sum(np.square(p2-p1)))

def theta(a: Point, b: Point, c: Point)-> float:
	v1, v2 = a-b, c-b
	return math.degrees(math.acos(np.dot(v1,v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))))

def find_hands(overlay: ndarray) -> None:
	frame_idx, latency = 0, 2
	fgmask, centroid = None, None
	overlay = cv2.resize(overlay, (250,200))
	min_frac, max_frac = 64, 4 # experimentally determined

	cap = cv2.VideoCapture(0)
	fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

	while(True):
	    ret, frame = cap.read()

	    if frame_idx % latency == 0: fgmask = fgbg.apply(frame)
	    frame_idx += 1

	    h, w = fgmask.shape


	    fgmask[:int(h*0.1), :] = 0
	    fgmask[int(h*0.9):, :] = 0
	    fgmask[:, int(w*0.4):int(w*0.6)] = 0

	    n_fingers = 0
	    image, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	    if contours:
		    lc = max(contours, key=lambda x: cv2.contourArea(x))

		    if (w*h//min_frac < cv2.contourArea(lc) < w*h//max_frac):

		    	hull = cv2.convexHull(lc, returnPoints=False)
		    	defects = cv2.convexityDefects(lc, hull)
		    	
		    	spts, epts, fpts, mpts = [], [], [], []
		    	for i in range(np.size(defects, 0)):
			    	spt = lc[defects[i,0,0]].flatten()
			    	ept = lc[defects[i,0,1]].flatten()
			    	fpt = lc[defects[i,0,2]].flatten()

			    	spts.append(spt)
			    	epts.append(ept)
			    	fpts.append(fpt)

			    	mpt = midpt(spt, ept)

			    	mpts.append(mpt)

			    	# Pretty lines
			    	# cv2.line(frame, tuple(spt), tuple(fpt), (0,255,0), 2)
			    	# cv2.line(frame, tuple(ept), tuple(fpt), (255,0,0), 2)

		    	centroid = compute_centroid(spts, fpts, epts)

		    	for mpt, fpt in zip(mpts, fpts):
			    	if pdst(mpt, fpt) < 25 \
			    	   or pdst(mpt, fpt) > 500 \
			    	   or mpt[1] > centroid[1] \
		    		   or not 0 <= abs(theta(centroid, mpt, fpt)) <= 30: continue
		    		n_fingers += 1
		    		
		    		# Pretty lines
		    		# cv2.line(frame, tuple(mpt), tuple(fpt), (0,255,0), 2)
			    	# cv2.line(frame, tuple(mpt), centroid, (0,0,255), 2)

		    	cv2.circle(frame, tuple(centroid), radius=8, color=(255, 0, 255), thickness=-1)

	    if n_fingers >= 3:
	    	try:
	    		cx, cy = tuple(centroid)
		    	oh, ow, _ = overlay.shape
	    		dx, dy = ow//2, oh//2
	    		bkgd = frame[cy-dy:cy+dy, cx-dx:cx+dx]
	    		frame[cy-dy:cy+dy, cx-dx:cx+dx] = np.where(overlay < 10, bkgd, overlay)
	    	except ValueError: print("\tOut of bounds")

	    cv2.imshow("img", frame)
	    if (cv2.waitKey(1) & 0xFF == ord('q')): break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	turkey = cv2.imread('cartoon.png')
	print("\tWe're live!")
	find_hands(turkey)

