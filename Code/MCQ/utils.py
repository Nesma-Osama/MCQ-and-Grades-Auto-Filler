# Imports
import cv2 as cv
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import math
from pytesseract import pytesseract
bubble_size=17
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

##########################################################################################
##paper extraction and wrap
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

# def ocr(image):
#     # Perform OCR
#     pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#     custom_config = r'--psm 6'
#     text = pytesseract.image_to_string(image, config=custom_config)
#     print(text)
########################################################################################
#  Fox crop the inner of the box in the paper
def cropp(image,contour):
        x, y, w, h = cv.boundingRect(contour)
        # Extract the ID box from the image
        new_image = image[y+25:y+h, x+10:x+w-10]
        return new_image    
def cropp_box_image(image, width, height, blur_image):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))  # Adjust kernel size
    denoised = cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations=4)
    
    # Find contours after morphological operations
    contours, _ = cv.findContours(denoised, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    inner_box = None
    
    for i in contours:
        area = cv.contourArea(i)
        if 0.05 * width * height < area < 0.9 * width * height:  # Area constraints
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4:  # If a quadrilateral is found
                inner_box = i
                break

    if inner_box is not None:
        # Crop the original and blurred images
        cropped_image = cropp(image, inner_box)
        cropped_blur_image = cropp(blur_image, inner_box)
        #######################################
        cropped_image[-5:, :] = 255  # Set last row of cropped_image to white
        cropped_blur_image[-5:, :] = 255  # Set last row of cropped_blur_image to white
        # Add a white box in all directions
        box_size = 10  # Width of the white border in pixels (adjust as needed)
        cropped_image = cv.copyMakeBorder(cropped_image, box_size, box_size, box_size, box_size, cv.BORDER_CONSTANT, value=255)
        cropped_blur_image = cv.copyMakeBorder(cropped_blur_image, box_size, box_size, box_size, box_size, cv.BORDER_CONSTANT, value=255)

        return cropped_image, cropped_blur_image
    else:
        # Add a white box to the original image
        # Add a white box to the original image
        image[-5:, :] = 255  # Set last row of cropped_image to white
        blur_image[-5:, :] = 255  # Set last row of cropped_blur_image to white
        ################################################
        box_size = 10  # Width of the white border in pixels (adjust as needed)
        image = cv.copyMakeBorder(image, box_size, box_size, box_size, box_size, cv.BORDER_CONSTANT, value=255)
        blur_image = cv.copyMakeBorder(blur_image, box_size, box_size, box_size, box_size, cv.BORDER_CONSTANT, value=255)

        return image, blur_image

####################################################################
## remove impluse noise        
def replace_image_with_white(contours,output,width):
    for contour in contours:       
        if cv.contourArea(contour) < width:
                cv.fillPoly(output, [contour], 255)
####################################################################3
#### extract name id mcq
def finall_extract(image,gray):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 5))  # Adjust kernel size
    denoised = cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations=4)
    # Find contours after open
    id_mcq_contours,_=cv.findContours(denoised,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    ###
    id_contour=None
    name_contour=None
    max_w=0
    mcq_regions = []  # List to store regions along with their x-coordinates
    for i in id_mcq_contours:
        x, y, w, h = cv.boundingRect(i)  # x, y: top-left corner, w: width, h: height
        if(w<0.9*image.shape[0] ):
            if(w>20 and w<100):###mcq
                    mcq_region=gray[y:y+h,x:x+w] 
                    mcq_regions.append(((x,y),mcq_region))
                    print(w)
            else : 
                if  w>max_w: 
                    max_w=w
                    name_contour=id_contour
                    id_contour=gray[y:y+h,x:x+w] 
                    
     # Sort the MCQ regions by x-coordinate
    mcq_regions_sorted = sorted(mcq_regions, key=lambda item: item[0][0])
    new_region=[mcq_regions_sorted[0][1]]
    # Display sorted MCQ regions
    for idx,( (x,y), mcq) in enumerate(mcq_regions_sorted):
        if( idx!=0 and np.abs(y-mcq_regions_sorted[0][0][0])>30):
            new_region.append( mcq)
    return new_region,id_contour,name_contour        
#################################################################################
def correct_id_mcq(image,list):
    image=cv.resize(image,(300,300))
    _, binary = cv.threshold(image, 160, 255, cv.THRESH_BINARY_INV)
    #kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 1))  # Adjust kernel size
    #denoised = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=3)
    contours,_=cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    n_rows=len(contours)
    print(n_rows)
    rows=np.array_split(binary,n_rows,axis=0)
    image_color = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    cv.drawContours(image_color, contours, -1, (0, 255, 0), 1)
    cv.imshow("contours",binary)
    # for i,row in enumerate(rows):
    #     cv.imshow(f"{i}",row)
    cv.waitKey(0)
    cv.destroyAllWindows()
##########################################
##split
def split_answers_from_row(row_image):
    # Preprocess the row image
    _, binary_row = cv.threshold(row_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
     # Find contours in the thresholded binary image
      # Optional morphological operations
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))  # Adjust kernel size
    binary = cv.morphologyEx(binary_row, cv.MORPH_CLOSE, kernel)
    #cv.imshow("black",binary)
    cv.waitKey(0)
    cv.destroyAllWindows()
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Sort contours by horizontal position
    contours = sorted(contours, key=lambda c: cv.boundingRect(c)[0])
    #print(len(contours))
    # Analyze intensity of each bubble
   # Analyze intensity of each bubble
    answer_parts = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        # Crop the bubble region from the original grayscale image
        bubble_region = row_image[y:y+h, x:x+w]

        # Calculate the mean intensity of the bubble region
        mean_intensity = cv.mean(bubble_region)[0]

        # Append the mean intensity to the answer list
        answer_parts.append(mean_intensity)
    #print("anser",answer_parts)   
    return answer_parts

        
def split_questions(image):
    # Preprocess the image
    padded_image = cv.copyMakeBorder(
        image, 
        20, 
        20, 
        20, 
        20, 
        borderType=cv.BORDER_CONSTANT, 
        value=255  # Padding color (255 for white, 0 for black)
    )
    # Apply thresholding after padding
    _, binary = cv.threshold(padded_image, 130, 255, cv.THRESH_BINARY_INV)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 3))
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=2)
    # Find contours
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Sort contours by vertical position (y-axis)
    contours = sorted(contours, key=lambda c: cv.boundingRect(c)[1])
    print(len(contours))
    # Group contours into rows based on their y-coordinate
    # Split the image into parts based on rows
    question_parts = []
    for contour in contours:  # Iterate over each contour in the row
            x, y, w, h =cv.boundingRect( contour)  # Contour bounding box
            # Crop the individual contour
            question_part = padded_image[y-5:y+h+5, x-5:x+w+5]        
            answers = split_answers_from_row(question_part)
            question_parts.append(answers)  
       
    return question_parts
    