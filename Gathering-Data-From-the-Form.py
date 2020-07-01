import cv2 
import numpy as np

I = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

#Load the dictionary that was used to generate the markers.
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

parameters =  cv2.aruco.DetectorParameters_create()
        
# Detect the markers in the image
markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(I, dictionary, parameters=parameters) 
# print(markerCorners)

detectedMarkers = I.copy()

cv2.aruco.drawDetectedMarkers(detectedMarkers, markerCorners, markerIds)

# cv2.imshow('out', detectedMarkers)
# cv2.waitKey()