# Imports
import cv2 as cv
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import math
from pytesseract import pytesseract

# Show Images

def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


# Find Biggest function
def find_biggest_contour(contours):
    biggest=np.array([])
    max_area=0
    for i in contours:
        area=cv.contourArea(i)
        if area>5000:
            peri=cv.arcLength(i,True)
            approx=cv.approxPolyDP(i,0.02*peri,True)
            if area>max_area and len(approx)==4:
                biggest=approx # 4 points
                max_area=area
    return biggest,max_area            
            
    
# Reorder points
def reorder(pts):
    # Sort the points based on their x and y coordinates
    pts = pts.reshape((4, 2))
    pts = sorted(pts, key=lambda x: x[0] + x[1])
    pts = np.array(pts, dtype="float32")
    # Top-left, top-right, bottom-right, bottom-left
    # We will reorder the points such that:
    # the first one will be top-left, second will be top-right, third will be bottom-right, and fourth will be bottom-left.
    rect = np.zeros((4, 2), dtype="float32")

    # Compute the sum and difference of the points to identify top-left and bottom-right
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect

def wrapped_paper(width,height,points,image):
    target_points = np.array([
        [0, 0],  # top-left
        [width - 1, 0],  # top-right
        [width - 1, height - 1],  # bottom-right
        [0, height - 1]  # bottom-left
    ], dtype="float32")
    ordered_points=reorder(points)
    # Get the perspective transform matrix
    matrix = cv.getPerspectiveTransform(ordered_points, target_points)

    # Warp the image
    warped_image = cv.warpPerspective(image, matrix, (width, height))
    return warped_image
        
# def if_has_outer_edge(contours, image):
#     has_outer_border = False  # Initialize as False
    
#     for contour in contours:
#         x, y, w, h = cv.boundingRect(contour)
#         print(w,h)
#         if w > 0.9 * image.shape[1] and h > 0.9 * image.shape[0]:
#             print("Outer border detected")
#             has_outer_border = True
#             break  # Stop checking further contours as we've found the outer border
    
#     if not has_outer_border:
#         print("No outer border detected")
    
#     return has_outer_border
       
def extract_id_box(adaptive_thresh,image):
    # Erode
    kernel = np.ones((5,5 ), np.uint8)  # Adjust kernel size based on the gaps
    erode = cv.erode(adaptive_thresh, kernel, iterations=1)
    # Find contours
    contours, _ = cv.findContours(erode, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    # Step 2: Filter contours based on aspect ratio and area
    id_contour = None
    min_area = 0
    for contour in contours:
        # Get bounding rectangle for each contour
        x, y, w, h = cv.boundingRect(contour)
        
        # Calculate aspect ratio (width / height)
        aspect_ratio = float(w) / h
        
        # Calculate area of the contour
        area = cv.contourArea(contour)
        
        # Filter contours by aspect ratio and area
        if 2.5 <= aspect_ratio <= 4 and area > min_area:
            min_area = area
            id_contour = contour  # Save the contour with the largest area and valid aspect ratio
            
    # If we found a valid ID contour, extract the ID box
    if id_contour is not None:
        # Get the bounding box of the largest valid contour
        x, y, w, h = cv.boundingRect(id_contour)
        # Extract the ID box from the image
        id_box = image[y:y+h, x:x+w]
        return id_box
    else:
            print("No ID box found")
            return None  # Return None if no valid ID box was found
# def ocr(image):
   
#     # Perform OCR
#     pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#     custom_config = r'--psm 6'
#     text = pytesseract.image_to_string(image, config=custom_config)
#     print(text)
    

