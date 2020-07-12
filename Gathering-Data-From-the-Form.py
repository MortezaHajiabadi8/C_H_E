import cv2 
import numpy as np
from glob import glob
from math import ceil, floor

def detectForm(src):
        
    #Load the dictionary that was used to generate the markers.
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    parameters =  cv2.aruco.DetectorParameters_create()
            
    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(src, dictionary, parameters=parameters) 

    detectedMarkers = src.copy()

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


    form = cv2.warpPerspective(src,H,  (rawForm.shape[1],rawForm.shape[0]))

    # cv2.imshow("form", form)
    # cv2.waitKey()
    return form

def cropForm(src):
    croped_form = src[110:600, :410]
    # cv2.imshow("cropped", croped_form)
    # cv2.waitKey()
    return croped_form

def thresholdedForm(src):
    #apply threshoding
    threshold = 90
    ret , T = cv2.threshold(src,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # cv2.imshow('Thresholded', T)
    # cv2.waitKey()
    return T
 
def find_contours(src):
    _, contours, _ = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def find_boxes_and_checkboxes(croped_form,contours):
    img2 = cv2.cvtColor(croped_form, cv2.COLOR_GRAY2BGR)
    my_contours = []
    for cnt in contours : 
        area = cv2.contourArea(cnt) 
        if 220 < area < 25000: 
            approx = cv2.approxPolyDP(cnt,  
                                    0.009 * cv2.arcLength(cnt, True), True) 
    
            if(len(approx) >= 4):  
                my_contours.append(cnt)
    
    top_left_corner = []
    boxes = []
    checkboxes = []
    box_names = ["ID", "FN", "LN"]        
    checkbox_names = ["PHD", "MS", "BS"]
    for cnt in my_contours:
        x,y,w,h = cv2.boundingRect(cnt)
        tolerance = [[x+t1,y+t2] for t1 in range(-6,7) for t2 in range(-6,7)]
        if  not [i for i in top_left_corner if i in tolerance]:
            top_left_corner.append([x,y])
            area = cv2.contourArea(cnt) 
            if area > 1000:
                boxes.append([cnt,y])
            else:
                checkboxes.append([cnt,x])
            
            
    boxes = sorted(boxes, key=lambda x:x[1])
    boxes = [[boxes[i][0],box_names[i]] for i in range(len(boxes))]
  
    checkboxes = sorted(checkboxes, key=lambda x:x[1])
    checkboxes = [[checkboxes[i][0],checkbox_names[i]] for i in range(len(checkboxes))]
    
    for cnt in boxes+checkboxes:
        x,y,w,h = cv2.boundingRect(cnt[0])
        cv2.rectangle(img2, (x,y), (x+w,y+h), (0,255,255), 2)
        cv2.imshow('image2', img2) 
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
        
    
    return boxes, checkboxes

def write_image_of_boxes(croped_form,boxes):
    image_of_boxes = []
    for box in boxes:
        x,y,w,h = cv2.boundingRect(box[0])
        dst_points = np.array([(0,0),
                            (w/8,0),
                            (w/8,h),
                            (0,h)], dtype=np.float32)
        for i in range(8):
                src_points = np.array([(x+i*((w-6)/8),y), (x+(i+1)*((w-6)/8),y), (x+(i+1)*((w-6)/8),y+h), (x+i*((w-6)/8),y+h)], dtype=np.float32)
                H = cv2.getPerspectiveTransform(src_points, dst_points)
                pic = cv2.warpPerspective(croped_form,H,  (h,ceil(w/8)))
                image_of_boxes.append(pic)
                cv2.imwrite(box[1]+str(i+1)+".jpg", pic)
                cv2.imshow(box[1]+str(i+1), pic)
                key = cv2.waitKey(0) & 0xFF
                if key != ord('q'):
                    cv2.destroyAllWindows()
                elif key == ord('q'):  
                    cv2.destroyAllWindows() 
                    break 

def write_image_of_checkboxes(croped_form, checkboxes):
    image_of_checkboxes = []
    for checkbox in checkboxes:
        x,y,w,h = cv2.boundingRect(checkbox[0])
        dst_points = np.array([(0,0),
                            (w,0),
                            (w,h),
                            (0,h)], dtype=np.float32)
        src_points = np.array([(x,y), (x+w,y), (x+w,y+h),(x,y+h)], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src_points, dst_points)
        pic = cv2.warpPerspective(croped_form,H,  (h,w))
        cv2.imwrite(checkbox[1]+".jpg", pic)
        key = cv2.waitKey(0) & 0xFF
        if key != ord('q'):
            cv2.destroyAllWindows()
        elif key == ord('q'):  
            cv2.destroyAllWindows() 
            break 
        
def concat_tile(im_list_2d):
    return np.concatenate([np.concatenate(im_list_h, axis=1) for im_list_h in im_list_2d], axis=0)


def to_matrix(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]
    
def main():
        I = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
        form = detectForm(I)
        croped_form = cropForm(form)
        thresholded_form = thresholdedForm(croped_form)
        contours = find_contours(thresholded_form)
        boxes, checkboxes = find_boxes_and_checkboxes(croped_form, contours)
        write_image_of_boxes(croped_form, boxes)
        write_image_of_checkboxes(croped_form, checkboxes)
    
        

if __name__ == '__main__':
    main()