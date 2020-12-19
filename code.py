import cv2
import numpy as np
from find_puzzle import find_puzzle, extract_digits
import imutils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json
from dlxsudoku import Sudoku
import h5py
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

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


# Loading Model from storage
#####################################################################
json_file = open("model3.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model3.h5")

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
#####################################################################


print("Processing image")
image = cv2.imread("pic3.jpeg")
image = imutils.resize(image, width=600)

(puzzleImage, warped) = find_puzzle(image, False)

board = np.zeros((9,9), dtype="int")
stepX = warped.shape[1]//9
stepY = warped.shape[0]//9

# print(stepX, stepY)

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
		digit=None
		try:
			digit = extract_digits(cell, False)
		except:
			pass

		if(digit is not None):
			# print("found 1 digit in cell {0},{1}".format(y,x))
			roi = cv2.resize(digit, (28,28))
			roi = roi.astype("float")/255
			roi = img_to_array(roi)
			roi = np.expand_dims(roi, axis=0)

			pred = model.predict(roi).argmax(axis=1)[0]
			board[y,x] = pred
			print(pred)

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
		print(print_sudoku(string_to_board(''.join(sudoku_string))))
	else:
		print(''.join(sudoku_string))
		break

try:
	sudoku_string = ''.join(sudoku_string)
	# sudoku_string="700000009009804500300702008007503200030020050008609400100906004004208100800000005"
	sudoku = Sudoku(sudoku_string)
	sudoku.solve()
	solved = str(sudoku.to_oneliner())
	solved_board = string_to_board(solved)

	print("\n\nPrinting solved board")
	print_sudoku(solved_board)

	plt.subplot(1,2,1)
	plt.imshow(puzzleImage)


	for (cellRow, boardRow) in zip(cellLocs, solved_board):
		# loop over individual cell in the row
		for (box, digit) in zip(cellRow, boardRow):
			# unpack the cell coordinates
			startX, startY, endX, endY = box
			# compute the coordinates of where the digit will be drawn
			# on the output puzzle image
			textX = int((endX - startX) * 0.33)
			textY = int((endY - startY) * -0.2)
			textX += startX
			textY += endY

			cell = warped[startY:endY, startX:endX]
			digi = extract_digits(cell, False)

			if(digi is None):
				# draw the result digit on the Sudoku puzzle image
				cv2.putText(puzzleImage, str(digit), (textX, textY),
					cv2.FONT_HERSHEY_SIMPLEX, 0.9, (98, 46, 32), 2)
	# show the output image
	
	plt.subplot(1,2,2)
	plt.imshow(puzzleImage)
	plt.show()

	cv2.imshow("Sudoku Result", puzzleImage)
	cv2.waitKey(0)
except:
	print("Sudoku solution not possible.")