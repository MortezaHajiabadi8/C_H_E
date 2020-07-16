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

def compute_dest_points():
    height = 700
    width = 500
    dest_points = np.array([(0,0),
                           (width,0),
                           (width,height),
                           (0,height)], dtype=np.float32)
    return dest_points, height, width

def detectForm(src, source_points, dest_points, height, width):
    H = cv2.getPerspectiveTransform(source_points, dest_points)
    detected_form = cv2.warpPerspective(src,H,  (width, height))
    cv2.imshow("detected form", detected_form)
    cv2.waitKey()
    return detected_form
  
def cropForm(detected_form):
    croped_form = detected_form[200:500, :370]
    cv2.imshow("cropped", croped_form)
    cv2.waitKey()
    return croped_form  

    
def main():
    I = cv2.imread("image.jpg")
    markerCorners, markerIds = detectMarkers(I)
    source_points = compute_source_points(markerCorners, markerIds)
    dest_points, height, width = compute_dest_points()
    detected_form = detectForm(I, source_points, dest_points, height, width)
    croped_form = cropForm(detected_form)
            

if __name__ == '__main__':
    main()