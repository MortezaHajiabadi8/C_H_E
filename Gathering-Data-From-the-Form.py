import cv2 
import numpy as np

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

cv2.imshow('form', form)

cv2.waitKey()