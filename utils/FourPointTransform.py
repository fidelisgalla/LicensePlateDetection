#thanks to pyimagesearch https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

import numpy as np
import cv2

# four point transform
def four_point_transform(image, pts):
	dst = np.array([[0, 0], [210, 0], [210, 50], [0, 50]], dtype=np.float32)
	#pts = np.array([[pts[0][0],pts[0][1],
	#				  pts[1][0],pts[1][1],
	#				  pts[2][0],pts[2][1],
	#				  pts[3][0],pts[3][1]]],dtype = np.float32)
	pts = np.array(pts,dtype = np.float32)

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(pts, dst)
	warped = cv2.warpPerspective(image, M, (200, 50))  #please change the value of 200 and 50 to respective value
	# return the warped image
	return warped