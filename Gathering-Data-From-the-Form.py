import cv2 
import numpy as np
from glob import glob
from math import ceil, floor

def detectMarkers(src):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters =  cv2.aruco.DetectorParameters_create()
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(src, dictionary, parameters=parameters) 
    detectedMarkers = src.copy()
    cv2.aruco.drawDetectedMarkers(detectedMarkers, markerCorners, markerIds)
    # cv2.imshow('Markers', detectedMarkers)
    # cv2.waitKey()
    return markerCorners, markerIds

def compute_source_points(markerCorners, markerIds):
    d = dict(enumerate(markerIds.flatten(), 1))
    key_list = list(d.keys())
    val_list = list(d.values()) 

    #33,34,35,36 are id's of markers in the form
    source_points = np.array([markerCorners[key_list[val_list.index(34)]-1][0][0],
                        markerCorners[key_list[val_list.index(35)]-1][0][1],
                        markerCorners[key_list[val_list.index(36)]-1][0][2],
                        markerCorners[key_list[val_list.index(33)]-1][0][3]], dtype=np.float32)
    
    return source_points 


    
    

    
def main():
    I = cv2.imread("image.jpg")
    markerCorners, markerIds = detectMarkers(I)
    source_points = compute_source_points(markerCorners, markerIds)
    
        
        
            

if __name__ == '__main__':
    main()