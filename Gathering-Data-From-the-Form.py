import cv2 
import numpy as np
import pytesseract

def sort_contours(cnts, method="left-to-right"):    
    # initialize the reverse flag and sort index
    reverse = False
    i = 0    
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True    
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1   
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][i], reverse=reverse))   
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def addPoints(p,q):
    m = p[0]+q[0]
    n = p[1]+q[1]
    return (m,n)

I = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

#Load the dictionary that was used to generate the markers.
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

parameters =  cv2.aruco.DetectorParameters_create()
        
# Detect the markers in the image
markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(I, dictionary, parameters=parameters) 

detectedMarkers = I.copy()

cv2.aruco.drawDetectedMarkers(detectedMarkers, markerCorners, markerIds)

# cv2.imshow('out', detectedMarkers)
# cv2.waitKey()

rawForm = cv2.imread('form.png', cv2.IMREAD_GRAYSCALE)

dest_points = np.array([(0,0),
                        (rawForm.shape[1],0),
                        (rawForm.shape[1],rawForm.shape[0]),
                        (0,rawForm.shape[0])], dtype=np.float32)


d = dict(enumerate(markerIds.flatten(), 1))
key_list = list(d.keys())
val_list = list(d.values()) 

src_points = np.array([markerCorners[key_list[val_list.index(34)]-1][0][0],
                       markerCorners[key_list[val_list.index(35)]-1][0][1],
                       markerCorners[key_list[val_list.index(36)]-1][0][2],
                       markerCorners[key_list[val_list.index(33)]-1][0][3]], dtype=np.float32)

# compute homography from point correspondences
H = cv2.getPerspectiveTransform(src_points, dest_points)


form = cv2.warpPerspective(I,H,  (rawForm.shape[1],rawForm.shape[0]))

# cv2.imshow("form", form)
# cv2.waitKey()

#crop form to delete dictionaries
croped_form = form[110:600, :]
# cv2.imshow("cropped", croped_form)
# cv2.waitKey()

#invert form to extract rectangle
inverted_form = 255 - croped_form

# cv2.imshow('inverted_form', inverted_form)
# cv2.waitKey()

#apply threshoding
threshold = 90
ret , T = cv2.threshold(inverted_form,threshold,255,cv2.THRESH_BINARY)

# cv2.imshow('Thresholded', T)
# cv2.waitKey()

#following lines are for find lines in image and then boxes

# Length(width) of kernel as 100th of total width
kernel_len = np.array(inverted_form).shape[1]//100
# Defining a vertical kernel to detect all vertical lines of image 
ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
# Defining a horizontal kernel to detect all horizontal lines of image
hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
# A kernel of 2x2
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))