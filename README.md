# Gathering-Data-From-the-Form

requirements:

    python 3.7.7 or below
    opencv-python==3.4.2.17 or below
    numpy

to see detected Markers in source image, uncomment the following lines:

    cv2.imshow('out', detectedMarkers)
    cv2.waitKey()

to see form after apply perspective on rawform, uncomment the following lines:

    cv2.imshow('form', form)
    cv2.waitKey()