import cv2 
import numpy as np

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
    #crop form to delete dictionaries
    croped_form = src[110:600, :]
    # cv2.imshow("cropped", croped_form)
    # cv2.waitKey()
    return croped_form

def invertForm(src):
    #invert form to extract rectangle
    inverted_form = 255 - src

    # cv2.imshow('inverted_form', inverted_form)
    # cv2.waitKey()
    return inverted_form

def thresholdedForm(src):
    #apply threshoding
    threshold = 90
    ret , T = cv2.threshold(src,threshold,255,cv2.THRESH_BINARY)

    # cv2.imshow('Thresholded', T)
    # cv2.waitKey()
    return T


def find_lines(inverted, thresholded):
    #following lines are for find lines in image and then boxes

    # Length(width) of kernel as 100th of total width
    kernel_len = np.array(inverted).shape[1]//100
    # Defining a vertical kernel to detect all vertical lines of image 
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    #Use vertical kernel to detect and save the vertical lines in a jpg
    image_1 = cv2.erode(thresholded, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
    # cv2.imshow("vertical", vertical_lines)
    # cv2.waitKey()

    #Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(thresholded, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
    # cv2.imshow("horizontal", horizontal_lines)
    # cv2.waitKey()

    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)#Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow("img_vh", img_vh)
    # cv2.waitKey()
    return img_vh

def find_contours(src):
    # Detect contours for following box detection
    _, contours, _ = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort all the contours by top to bottom.
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")
    return contours

def find_fields_and_boxes(contours):
    #find big contours to detect fields and checkboxes
    areas = []
    for c in contours:
        areas.append(cv2.contourArea(c))
    areas = list(enumerate(areas))
    areas = sorted(areas, key=lambda x: x[1])
    areas = areas[::-1]

    #index of big countors are 1,2,3,7,8,9 in areas
    #index of big field are 1,2,3
    #index of checkbox are 10,11,12

    big_fileds = [1,2,3]
    #assign Appropriate name to fields by use of coordinates of points
    big_fileds_points = []
    for i in big_fileds:
        j = areas[i][0]
        cnt = contours[j]
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
        big_fileds_points.append(approx[0][0][1])
    big_fileds_points = zip(big_fileds,big_fileds_points)
    big_fileds_points = sorted(big_fileds_points, key=lambda x: x[1])
    big_fileds_names = ["ID", "FN", "LN"]
    for i in range(3):
        big_fileds_points[i] = list(big_fileds_points[i])
        big_fileds_points[i].append(big_fileds_names[i])
    big_fileds = big_fileds_points.copy() #[[2, 125, 'ID'], [3, 188, 'FN'], [1, 247, 'LN']]

    #assign Appropriate name to checkboxes by use of coordinates of points
    checkboxes = [7,8,9]
    checkbox_points = []
    for i in checkboxes:
        j = areas[i][0]
        cnt = contours[j]
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
        checkbox_points.append(approx[0][0][0])

    checkbox_points = zip(checkboxes,checkbox_points)
    checkbox_points = sorted(checkbox_points, key=lambda x: x[1])
    checkboxes_names = ["PHD", "MS", "BS"]
    for i in range(3):
        checkbox_points[i] = list(checkbox_points[i])
        checkbox_points[i].append(checkboxes_names[i])
    checkboxes = checkbox_points.copy() #[[7, 55, 'PHD'], [8, 159, 'MS'], [9, 307, 'BS']]

    return big_fileds,checkboxes, areas

def detect_and_write_boxes_to_file(checkboxes, areas, contours, croped_form):
    for contour in checkboxes:
        j = areas[contour[0]][0]
        cnt = contours[j]
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
        p1 = (approx[0][0][0], approx[0][0][1])
        p4 = (approx[7][0][0], approx[7][0][1])
        width = approx[3][0][0]-approx[0][0][0]
        height = approx[7][0][1]-approx[0][0][1]
        p2 = (p1[0]+width, p1[1])
        p3 = (p1[0]+width, p1[1]+height)
        p4 = (p1[0], p1[1]+height)
        source__points = np.array([p1,p2,p3,p4], dtype=np.float32)
        dst_points = np.array([(0,0),
                            (width,0),
                            (width,height),
                            (0,height)], dtype=np.float32)
        H = cv2.getPerspectiveTransform(source__points, dst_points)
        pic = cv2.warpPerspective(croped_form,H,  (height,width))
        cv2.imwrite(contour[2]+".jpg", pic)
        # cv2.imshow(contour[2], pic)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

def detect_and_write_fileds_to_file(big_fields, areas, contours, croped_form):
    for contour in big_fields:
        
        j = areas[contour[0]][0]
        cnt = contours[j]
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
        width = approx[1][0][0]-approx[0][0][0]
        height = approx[3][0][1]-approx[0][0][1]
        w = width//8
        up = (approx[0][0][0],approx[0][0][1])
        down = (approx[3][0][0],approx[3][0][1])
        
        dst_points = np.array([(0,0),
                            (w,0),
                            (w,height),
                            (0,height)], dtype=np.float32)
        for k in range(8):
            tempup1 = (k*w,0)
            tempup2 = ((k+1)*w,0)
            tempdown1 =(k*w,0)
            tempdown2 =((k+1)*w,0)
            p1 = addPoints(up,tempup1)
            p2 = addPoints(up,tempup2)
            p3 = addPoints(down, tempdown2)
            p4 = addPoints(down, tempdown1)
            source__points = np.array([p1,p2,p3,p4], dtype=np.float32)
            H = cv2.getPerspectiveTransform(source__points, dst_points)
            pic = cv2.warpPerspective(croped_form,H,  (height,w))
            cv2.imwrite(contour[2]+str(k+1)+".jpg", pic)
            # cv2.imshow(c[2]+str(k+1), pic)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

def main():
    I = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    form = detectForm(I)
    croped_form = cropForm(form)
    inverted_form = invertForm(croped_form)
    thresholded_form = thresholdedForm(inverted_form)
    img_vh = find_lines(inverted_form, thresholded_form)
    contours = find_contours(img_vh)
    fields, boxes, areas = find_fields_and_boxes(contours)
    detect_and_write_boxes_to_file(boxes, areas, contours, croped_form)
    detect_and_write_fileds_to_file(fields, areas, contours, croped_form)
    
    
if __name__ == '__main__':
    main()