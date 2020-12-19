from matplotlib import pyplot as plt
import numpy as np
import cv2
import imutils
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border

def find_puzzle(image, debug=False):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (7, 7), 3)
	# blurred = cv2.medianBlur(image, 5)
	thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	thresh = cv2.bitwise_not(thresh)

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	puzzleCnt = None

	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02*peri, True)
		if(len(approx)==4):
			puzzleCnt = approx
			break

	if(puzzleCnt is None):
		raise Exception("Could not find Sudoku puzzle outline.")

	puzzle = four_point_transform(image, puzzleCnt.reshape(4,2))
	warped = four_point_transform(gray, puzzleCnt.reshape(4,2))

	if(debug):
		output = image.copy()
		cv2.drawContours(output, [puzzleCnt], -1, (0,255,0),2)
		plt.subplot(1,3,1)
		plt.imshow(output)
	
		plt.subplot(1,3,2)
		plt.imshow(puzzle)

		plt.subplot(1,3,3)
		plt.imshow(warped)
		plt.show()

	return (puzzle, warped)

def extract_digits(cell, debug=False):
	thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	thresh = clear_border(thresh)

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	if(len(cnts)==0):
		return None

	c = max(cnts, key=cv2.contourArea) 
	mask = np.zeros(thresh.shape, dtype="uint8")
	cv2.drawContours(mask, [c], -1, 255, -1)

	(h, w) = thresh.shape
	percentFilled = cv2.countNonZero(mask)/float(w*h)

	if(percentFilled<0.05):
		return None

	digit = cv2.bitwise_and(thresh, thresh, mask=mask)

	if(debug):
		plt.subplot(1,2,1)
		plt.imshow(thresh)
		plt.subplot(1,2,2)
		plt.imshow(digit)
		plt.show()

	return digit


