import cv2
import numpy as np

def bestMatch(contours):
    index = 0; best = 0; areaMax =0;
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx)>8:
            area = cv2.contourArea(contour);
            if area >= areaMax:
                areaMax = area
                best=index
        index+=1
    return best

def closingLines(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    image_dil = cv2.dilate(image, kernel, iterations=1)
    return cv2.erode(image_dil, kernel, iterations=1)

def cropImage(image, contours, index_of_contour):
    mask = np.zeros_like(image)
    #mask = cv2.drawContours(mask, contours, index_of_contour, 255, -1)
    (x,y),radius = cv2.minEnclosingCircle(contours[index_of_contour])
    center = (int(x), int(y))
    mask =cv2.circle(mask, center, int(radius), 255, -1)
    mask[:, :, 1] = mask[:,:,0]
    mask[:, :, 2] = mask[:, :, 0]
    ret_val = np.zeros_like(image)
    ret_val[mask == 255] = image[mask == 255]
    return ret_val

def findWBC(image):

    green_channel = image[:, :, 2]
    #izdvajamo najtamnije delove slike , jer su leukociti najtamniji
    ret, image_bin = cv2.threshold(green_channel, 150, 255, cv2.THRESH_BINARY)

    #nadji ivice od svih tamnih delova slike
    image_with_edges = cv2.Canny(image_bin, 5, 200)
    closed_image = closingLines(image_with_edges)


    img, contours, hierarchy = cv2.findContours(closed_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cropped_image = np.zeros_like(image)
    if len(contours) != 0:
        bestContourIndex = bestMatch(contours)
        cropped_image = cropImage(image, contours, bestContourIndex)

    return cropped_image
