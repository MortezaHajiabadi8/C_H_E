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

def thresholdForm(croped_form):
    croped_form = cv2.cvtColor(croped_form, cv2.COLOR_BGR2GRAY)
    ret , T = cv2.threshold(croped_form,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('Thresholded', T)
    cv2.waitKey()
    return T

def find_contours(src):
    _, contours, _ = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_contours_based_on_area_and_number_of_sides(croped_form, contours):
    out = croped_form.copy()
    contours_based_on_area_and_number_of_sides = []
    for contour in contours : 
        area = cv2.contourArea(contour) 
        if 220 < area < 25000: 
            approx = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), True) 
            number_of_sides = len(approx)
            if(number_of_sides >= 4):  
                contours_based_on_area_and_number_of_sides.append(contour)
                cv2.drawContours(out, [contour], 0, (0,255,255), 2)
                cv2.imshow('contours_based_on_area_and_number_of_sides', out)
                cv2.waitKey()
    cv2.destroyAllWindows()           
    return contours_based_on_area_and_number_of_sides

def find_boxes_and_checkboxes(croped_form, contours_based_on_area_and_number_of_sides):
    top_left_corner = []
    boxes = []
    checkboxes = []
    
    for contour in contours_based_on_area_and_number_of_sides:
        x,y,w,h = cv2.boundingRect(contour)
        tolerance = [[x+t1,y+t2] for t1 in range(-6,7) for t2 in range(-6,7)]
        if  not [i for i in top_left_corner if i in tolerance]:
            top_left_corner.append([x,y])
            area = cv2.contourArea(contour) 
            if area > 1000:
                boxes.append([contour,y])
            else:
                checkboxes.append([contour,x])
    
    return boxes, checkboxes

def assign_name_to_boxes(croped_form, boxes):
    box_names = ["ID", "FN", "LN"] 
    sorted_boxes = sorted(boxes, key=lambda x:x[1])
    boxes_with_name = [[sorted_boxes[i][0],box_names[i]] for i in range(len(sorted_boxes))]
    for box in boxes_with_name:
        out = croped_form.copy()
        cv2.drawContours(out, [box[0]], 0, (0,255,255), 2)
        cv2.imshow(box[1], out)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return boxes_with_name

def assign_name_to_checkboxes(croped_form, checkboxes):
    checkbox_names = ["PHD", "MS", "BS"]
    sorted_checkboxes = sorted(checkboxes, key=lambda x:x[1])
    checkboxes_with_name = [[sorted_checkboxes[i][0],checkbox_names[i]] for i in range(len(sorted_checkboxes))]
    for checkbox in checkboxes_with_name:
        out = croped_form.copy()
        cv2.drawContours(out, [checkbox[0]], 0, (0,255,255), 2)
        cv2.imshow(checkbox[1], out)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return checkboxes_with_name

def write_image_of_boxes(croped_form, boxes_with_name):
    for box in boxes_with_name:
        x,y,w,h = cv2.boundingRect(box[0])
        little_box_width = w/8
        dest_points = np.array([(0,0),(little_box_width,0),(little_box_width,h),(0,h)], dtype=np.float32)
        number_of_little_box = 8
        for i in range(number_of_little_box):
            sourc_points = np.array([(x+i*(little_box_width-1),y), (x+(i+1)*(little_box_width-1),y), (x+(i+1)*(little_box_width-1),y+h), (x+i*(little_box_width-1),y+h)], dtype=np.float32)
            H = cv2.getPerspectiveTransform(sourc_points, dest_points)
            pic = cv2.warpPerspective(croped_form,H,  (h,ceil(w/8)))
            cv2.imwrite(box[1]+str(i+1)+".jpg", pic)

def write_image_of_checkboxes(croped_form, checkboxes_with_name):
    for checkbox in checkboxes_with_name:
        x,y,w,h = cv2.boundingRect(checkbox[0])
        dest_points = np.array([(0,0),(w,0),(w,h),(0,h)], dtype=np.float32)
        source_points = np.array([(x,y), (x+w,y), (x+w,y+h),(x,y+h)], dtype=np.float32)
        H = cv2.getPerspectiveTransform(source_points, dest_points)
        pic = cv2.warpPerspective(croped_form,H,  (h,w))
        cv2.imwrite(checkbox[1]+".jpg", pic)
        
        
def main():
    I = cv2.imread("image.jpg")
    markerCorners, markerIds = detectMarkers(I)
    source_points = compute_source_points(markerCorners, markerIds)
    dest_points, height, width = compute_dest_points()
    detected_form = detectForm(I, source_points, dest_points, height, width)
    croped_form = cropForm(detected_form)
    thresholded_form = thresholdForm(croped_form)  
    contours = find_contours(thresholded_form)
    contours_based_on_area_and_number_of_sides = find_contours_based_on_area_and_number_of_sides(croped_form, contours)
    boxes, checkboxes = find_boxes_and_checkboxes(croped_form, contours_based_on_area_and_number_of_sides)
    boxes_with_name = assign_name_to_boxes(croped_form, boxes)
    checkboxes_with_name = assign_name_to_checkboxes(croped_form, checkboxes)
    write_image_of_boxes(croped_form, boxes_with_name)
    write_image_of_checkboxes(croped_form, checkboxes_with_name)
    
if __name__ == '__main__':
    main()