# IMPORT NUMPY AND CV2
import numpy as np # importing numpy, a library that allows for complex data structures
import cv2 # importing opencv, a image processing library for python
import imutils

# VIDEO = VIDEO CAPTURE
vid = cv2.VideoCapture(0) # setting vid equal to index 0 capture (default webcam)

# WHILE TRUE
while (True):

    # IMAGE PROCESSING
    ret, img = vid.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (3, 3))
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    # INITIALIZE VARS FOR MASK
    x = 400
    y = 300
    w = 550
    h = 575
    mask = np.zeros(edges.shape[:2], np.uint8)
    mask[y:y + h, x:x + w] = 255
    maskimg = cv2.bitwise_and(edges, edges, mask=mask)

    heirarchy, contours = cv2.findContours(maskimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours((contours, heirarchy))

    if contours is not None:

        cnt = contours[0]
        ctr = np.array(cnt).reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(img, [ctr], -1, (0, 255, 0), 3)

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

        cv2.imshow('frame', img)

    else:
        cv2.imshow('frame', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()


# PROGRAMMING WORKS CITED

# Video capture and video display (Lines 1-12, 77-88) https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
# Hough Lines detection (Lines 28, 32, 39, 50) https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html
# Masking (Lines 18-25): https://stackoverflow.com/questions/11492214/opencv-via-python-is-there-a-fast-way-to-zero-pixels-outside-a-set-of-rectangle
# OpenCV rectangle (Line 75) https://www.geeksforgeeks.org/python-opencv-cv2-rectangle-method/
# OpenCV line (Lines 46, 72) https://www.geeksforgeeks.org/python-opencv-cv2-line-method/