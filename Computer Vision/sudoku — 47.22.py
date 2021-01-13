import joblib

import numpy as np
from numpy import logical_and as land
from numpy import logical_not as lnot
from skimage.feature import canny
from skimage.transform import rescale, ProjectiveTransform, warp
from skimage.morphology import dilation, disk
import cv2
import operator

# import train


def infer_grid(img):
    """Infers 81 cell grid from a square image."""
    squares = []
    side = img.shape[:1]
    side = side[0] / 9
    for i in range(9):
        for j in range(9):
            p1 = (i * side, j * side)  # Top left corner of a bounding box
            p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner of bounding box
            squares.append((p1, p2))
    return squares

def distance(p1, p2):
    """Returns the scalar distance between two points"""
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))

def predict(img,clf):
    img = img[25:125,25:125,]
    img = cv2.GaussianBlur(img,(3,3),0)
    img = img[..., ::-1]

    HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    LIGHT = HLS[:, :, 1]
    LIGHT = 255. - LIGHT
    LIGHT[LIGHT<150] = 0
    LIGHT = LIGHT / np.maximum(np.max(LIGHT), 1) * 255.
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(LIGHT,kernel,iterations = 1)
    
    if np.max(erosion) < 50:
        return -1
    else:
        pred_img = np.array([cv2.resize(erosion,(28,28)).ravel()])
        pred = clf.predict(pred_img)[0]
        return pred

def predict_image(img):
    # loading model:  (you can use any other pickle-like format)
    clf = joblib.load('/autograder/submission/random_forest.joblib')

    img_orig = img.copy()
    img = img[..., ::-1]

    img = cv2.GaussianBlur(img.copy(), (9, 9), 0)

    HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    LIGHT = HLS[:,:, 1]
    mask = (LIGHT < 100) | (LIGHT > 250)
    mask_int = mask.astype(np.uint8)
    
    ext_contours = cv2.findContours(mask_int.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    ext_contours = sorted(ext_contours, key=cv2.contourArea, reverse=True)
    contours = [cnt for cnt in ext_contours if cv2.contourArea(cnt) > cv2.contourArea(ext_contours[0])*0.7 ]
    
    im_mask = np.zeros(img_orig.shape[:2],dtype=np.uint8)
    digits = []
    for contour in contours:

        # Bottom-right point has the largest (x + y) value
        # Top-left has point smallest (x + y) value
        # Bottom-left point has smallest (x - y) value
        # Top-right point has largest (x - y) value
        bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in contour]), key=operator.itemgetter(1))
        top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in contour]), key=operator.itemgetter(1))
        bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in contour]), key=operator.itemgetter(1))
        top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in contour]), key=operator.itemgetter(1))
        
        crop_rect = [contour[top_left][0], contour[top_right][0], contour[bottom_right][0], contour[bottom_left][0]]
        
        # Rectangle described by top left, top right, bottom right and bottom left points
        top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

        # Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
        src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
        
        masks = np.array([[top_left, top_right, bottom_right, bottom_left]], dtype=np.int32)

        
        im_mask = cv2.fillPoly(im_mask, masks,255)
        
        
        # Get the longest side in the rectangle
        side = max([
            distance(bottom_right, top_right),
            distance(top_left, bottom_left),
            distance(bottom_right, bottom_left),
            distance(top_left, top_right)
        ])

        # Describe a square with side of the calculated length, this is the new perspective we want to warp to
        dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

        # Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
        m = cv2.getPerspectiveTransform(src, dst)

        # Performs the transformation on the original image
        normalized_image = cv2.warpPerspective(img_orig, m, (int(side), int(side)))
        
        sq = infer_grid(normalized_image)

        sudoku_imgs = []
        for i in range(len(sq)):
            x_top = int(sq[i][0][0])
            y_top = int(sq[i][0][1])
            
            x_b = int(sq[i][1][0])
            y_b = int(sq[i][1][1])
            
            sudoku_imgs.append(normalized_image[x_top:x_b,y_top:y_b])

        pred_imgs = [predict(sudoku_imgs[num],clf) for num,image in enumerate(sudoku_imgs)]
        
        digits.append(np.reshape(pred_imgs, (9, 9)).astype(np.int16))

    mask = np.array(im_mask.astype(bool))
    return mask, digits

def main():
    train_img = cv2.imread('/autograder/source/train/train_4.jpg')
 
    mask, sudoku_digits = predict_image(train_img)

    
main()