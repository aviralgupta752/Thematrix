import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json
from dlxsudoku import Sudoku
import h5py
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from skimage.segmentation import clear_border

def print_sudoku(board):
	print("-"*37)
	for i, row in enumerate(board):
		print(("|" + " {}   {}   {} |"*3).format(*[x if x != 0 else " " for x in row]))
		if i == 8:
			print("-"*37)
		elif i % 3 == 2:
			print("|" + "---+"*8 + "---|")
		else:
			print("|" + "   +"*8 + "   |")

def board_to_string(board):
	s = ""
	for i in range(9):
		for j in range(9):
			s+=str(board[i,j])

	return s

def string_to_board(s):
	board = []
	for i in range(9):
		l=[]
		for j in range(9):
			l.append(int(s[i*9+j]))
		board.append(l)

	return board

def read_image(image, debug=False):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# blurred = cv2.GaussianBlur(gray, (7, 7), 3)
	blurred = cv2.GaussianBlur(gray.copy(), (9, 9), 0)
	# blurred = cv2.medianBlur(image, 5)
	thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	thresh = cv2.bitwise_not(thresh, thresh)

	kernel = np.array([[0,1,0], [1,1,1], [0,1,0]], np.uint8)
 	process = cv2.dilate(thresh, kernel)

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

	puzzleCnt = None

	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.04*peri, True)
		if(len(approx)==4):
			puzzleCnt = approx
			break

	if(puzzleCnt is None):
		return None

	puzzle = four_point_transform(image, puzzleCnt.reshape(4,2))
	warped = four_point_transform(gray, puzzleCnt.reshape(4,2))

	if(debug):
		output = image.copy()
		cv2.drawContours(output, [puzzleCnt], -1, (0,255,0),2)
		cv2.imshow('output', output)
		cv2.imshow('puzzle',puzzle)
		cv2.waitKey(0)
	return (puzzle, warped)

def extract_digits(cell, debug=False):
	for j in range(10):
		thresh = cv2.threshold(cell, 20*j, 255, cv2.THRESH_BINARY_INV)[1]
		thresh = clear_border(thresh)

		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		
		if(len(cnts)==0):
			continue

		c = max(cnts, key=cv2.contourArea) 
		mask = np.zeros(thresh.shape, dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)

		(h, w) = thresh.shape
		percentFilled = cv2.countNonZero(mask)/float(w*h)

		if(percentFilled<0.05):
			continue

		digit = cv2.bitwise_and(thresh, thresh, mask=mask)
		if(debug):
			print("j =",j)
			cv2.imshow('thresh', thresh)
			cv2.imshow('digit',digit)
			cv2.waitKey(0)

		return digit

	return None

if __name__=="__main__":
	json_file = open("model6.json", 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	model.load_weights("model6.h5")

	model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

	(puzzle, warped) = read_image(cv2.imread("pic6.jpeg"), False)
	board = np.zeros((9,9), dtype="int")
	stepX = warped.shape[1]//9
	stepY = warped.shape[0]//9

	cellLocs = []

	for y in range(0,9):
		row=[]
		for x in range(0,9):
			startX = x*stepX
			startY = y*stepY
			endX = (x+1)*stepX
			endY = (y+1)*stepY

			row.append((startX, startY, endX, endY))
			cell = warped[startY:endY, startX:endX]

			l=[]
			for i in range(-15, 16,2):
				if(i in range(-12,13)):
					continue

				h,w = cell.shape[:2]
				T = np.float32([[1,0,w/(i*1.0)], [0,1,0]])
				shifted_cell = cv2.warpAffine(cell, T, (w,h))

				digit = None
				digit = extract_digits(shifted_cell)
				
				if(digit is not None):
					roi = cv2.resize(digit, (28,28))
					roi = roi.astype("float")/255
					roi = img_to_array(roi)
					roi = np.expand_dims(roi, axis=0)

					pred = model.predict(roi).argmax(axis=1)[0]
					l.append(pred)

			if(len(l)!=0):
				# print "list =", l
				board[y,x] = max(set(l), key = l.count) 
				# print(board[y,x])

		cellLocs.append(row)

	print("OCR'd Sudoku board: ")
	print_sudoku(board)

	sudoku_string = board_to_string(board)
	sudoku_string = list(sudoku_string)

	while(True):
		print("Enter option:")
		print("1. Replace value")
		print("2. Break")
		opt = int(raw_input())

		if(opt==1):
			r,c,value =map(int, raw_input("Enter i, j, value : ").split())
			sudoku_string[r*9+c]=str(value)
			print_sudoku(string_to_board(''.join(sudoku_string)))
		else:
			break

	try:
		sudoku_string = ''.join(sudoku_string)
		sudoku = Sudoku(sudoku_string)
		sudoku.solve()
		solved = str(sudoku.to_oneliner())
		solved_board = string_to_board(solved)
	
		plt.subplot(1,2,1)
		plt.imshow(puzzle)
		for (cellRow, boardRow) in zip(cellLocs, solved_board):
			for (box, digit) in zip(cellRow, boardRow):
				startX, startY, endX, endY = box
				textX = int((endX - startX) * 0.33)
				textY = int((endY - startY) * -0.2)
				textX += startX
				textY += endY

				cell = warped[startY:endY, startX:endX]
				digi = extract_digits(cell, False)

				if(digi is None):
					cv2.putText(puzzle, str(digit), (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (32, 46, 98), 2)
		# show the output image
		plt.subplot(1,2,2)
		plt.imshow(puzzle)
		plt.show()

	except:
	 	print("Error")
