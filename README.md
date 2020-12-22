# The Matrix
<h1> An Augmented Reality Sudoku </h1>

This program brings out the real fun in solving sudoku. Often the people who love to solve sudoku do a mistake and it spoils everything, moreover they have to wait for the next day newspaper to see where they did it wrong, but using our program they can get the solutions with one click. Just feed the image to the program and it will do the rest.
The program takes an image, removes noise from it by blurring, finds the sudoku board in it using opencv and skimage, divide the board into cells, identifies the digit in each cell using convolutional neural nets and shows the solution on the original image.

<h3> For installing all the libraries used: </h3>

pip install opencv-python

pip install numpy

pip install imutils

pip install --upgrade tensorflow

pip install dlxsudoku

pip install h5py

pip install matplotlib

pip install scikit-image


If any of the above pips doesn't work, please google the error.

For the most part of the project, I have referred to this link: https://www.pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/

The model has been trained on mnist digit recogniser dataset using tensorflow 2.1.0. You can directly download weights from this repo. This model has an accuracy of 98.5% apporximately. Sometimes, the program can recognise the digit incorrectly which inturn will result in no solution. To make it errorfree, I have provided the feature to correct incorrect digits. 

**This model works best if the sudoku is in center of the image.**

**Please change the image address before running the program.**

