import os
import cv2
import math
import argparse
import numpy as np
from typing import Any, Dict, List, NewType, Set, Tuple

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

def run(turkey_path: ndarray, turkey_dims: Tuple[int, int], latency: int, persistance: int,
		min_frac: int, max_frac: int, theta_max: float, min_n_fingers: int, quiet: bool)-> None:
	frame_idx = 0
	persistance_idx = 0
	tree  = cv2.RETR_TREE
	chain = cv2.CHAIN_APPROX_SIMPLE
	fgmask, centroid = None, None

	persisting = False

	overlay = cv2.imread(turkey_path)
	overlay = cv2.resize(overlay, turkey_dims)

	cap = cv2.VideoCapture(0)
	fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

	while(True):
		ret, frame = cap.read()

		try:
			if frame_idx % latency == 0: fgmask = fgbg.apply(frame)
			frame_idx += 1

			h, w = fgmask.shape

			fgmask[:int(h*0.1), :] = 0 # hardcoded for consistency,
			fgmask[int(h*0.9):, :] = 0 # but can be safely modified
			fgmask[:, int(w*0.4):int(w*0.6)] = 0

			n_fingers = 0 if not persisting else 5
			_, contours, _ = cv2.findContours(fgmask, tree, chain)

			if contours and not persisting:
				lc = max(contours, key=lambda x: cv2.contourArea(x))

				if (w*h//min_frac < cv2.contourArea(lc) < w*h//max_frac):

					hull = cv2.convexHull(lc, returnPoints=False) # must be False !!
					defects = cv2.convexityDefects(lc, hull)

					spts, epts, fpts, mpts = [], [], [], []
					for i in range(np.size(defects, 0)):
						spt = lc[defects[i,0,0]].flatten(); spts.append(spt)
						ept = lc[defects[i,0,1]].flatten(); epts.append(ept)
						fpt = lc[defects[i,0,2]].flatten(); fpts.append(fpt)

						mpts.append(midpt(spt, ept))

						# Pretty lines :)
						# cv2.line(frame, tuple(spt), tuple(fpt), (0,255,0), 2)
						# cv2.line(frame, tuple(ept), tuple(fpt), (255,0,0), 2)

					centroid = compute_centroid(spts, fpts, epts)

					for mpt, fpt in zip(mpts, fpts):
						if mpt[1] > centroid[1] \
						   or not 25 < pdst(mpt, fpt) < 500 \
						   or not 0 <= abs(theta(centroid, mpt, fpt)) <= 30: continue
						n_fingers += 1
						
						# Pretty lines :)
						# cv2.line(frame, tuple(mpt), tuple(fpt), (0,255,0), 2)
						# cv2.line(frame, tuple(mpt), centroid, (0,0,255), 2)
					if not quiet:
						cv2.circle(frame, tuple(centroid), radius=8, color=(255, 0, 255), thickness=-1)

			if n_fingers >= min_n_fingers: # 3 seems to be best
				try:
					persistance_idx -= 1
					cx, cy = tuple(centroid)
					oh, ow, _ = overlay.shape
					dx, dy = ow//2, oh//2
					bkgd = frame[cy-dy-100:cy+dy-100, cx-dx:cx+dx]
					frame[cy-dy-100:cy+dy-100, cx-dx:cx+dx] = np.where(overlay < 10, bkgd, overlay)
					if not quiet: print("\t ~ Gobble Gobble Gobble ~")
					if not persisting: persistance_idx = 4; persisting = True
					if not persistance_idx: persisting = False
				except ValueError:
					if not quiet: print("\tOut of bounds!!")
		except: print("\tWarning: Internal Error ")
		finally:
			cv2.imshow("img", frame)
			if (cv2.waitKey(1) & 0xFF == ord('q')): break

	cap.release()
	cv2.destroyAllWindows()

def get_hypers()-> Dict[str, Any]:
	parser = argparse.ArgumentParser()
	parser.add_argument("-l", "--latency", type=int, help="Skip every l-th frame", default=2)
	parser.add_argument("-t", "--turkey_path", type=str, help="Path to turkey image file", default="./cartoon.png")
	parser.add_argument("-d", "--turkey_dims", type=int, nargs=2, help="(h,w) of turkey", default=(350,300))
	parser.add_argument("-p", "--turkey_persistance", type=int, help="n_frames each turkey lasts for", default=5)
	parser.add_argument("-f", "--min_max_fracs", type=int, nargs=2, help="min and max contour area, \
													as a fraction of total frame area.", default=(64, 4))
	parser.add_argument("-a", "--max_angle", type=int, help="max acceptable angle to classify as finger", default=30)
	parser.add_argument("-m", "--min_n_fingers", type=int, help="min number of fingers to be considered a hand", default=3)
	parser.add_argument("-q", "--quiet", type=int, help="silence all output", default=0)
	args = vars(parser.parse_args())

	try:

		l = int(args["latency"])
		tp = args["turkey_path"]
		a = int(args['max_angle'])
		m = int(args["min_n_fingers"])
		p = int(args["turkey_persistance"])
		d = tuple(map(int, args["turkey_dims"]))
		lf,hf = tuple(map(int, args["min_max_fracs"]))
		q = True if args["quiet"] > 0 else False
		
		assert(0 <= m <= 5)
		assert(0 <  p <= 100)
		assert(0 <= a <= 360)
		assert((1 < lf < 100) and (1 < hf < 100))
		assert(os.path.isfile(tp))

	except AssertionError:
		print('\tOne or more of your inputs is out of bounds - please try again.')
		exit()
	except:
		print('\tOne or more of your inputs is invalid - please try again.')
		exit()

	return tp, d, l, p, lf, hf, a, m, q

if __name__ == '__main__':
	tp, d, l, p, lf, hf, a, m, q = get_hypers()
	print("\t[TURKEY MODE ACTIVATED]")
	run(tp, d, l, p, lf, hf, a, m, q)

