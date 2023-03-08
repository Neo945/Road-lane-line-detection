#Importing some useful packages
import numpy as np
import cv2

def convert_hsl(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

def HSL_color_selection(image):
    converted_image = convert_hsl(image)
    
    lower_threshold = np.uint8([0, 200, 0])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)
    
    lower_threshold = np.uint8([10, 0, 100])
    upper_threshold = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)
    
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask = mask)
    
    return masked_image

# Image to grey scale image
def gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Smoothing of image
def gaussian_smoothing(image, kernel_size = 13):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)   

# Using canny edge detection to detect edges of road
def canny_detector(image, low_threshold = 50, high_threshold = 150):
    return cv2.Canny(image, low_threshold, high_threshold)

# Masking reagion of interest
def region_selection(image):
    mask = np.zeros_like(image)   
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    rows, cols = image.shape[:2]
    # bottom_left  = [cols * 0.1, rows * 0.95]
    # top_left     = [cols * 0.4, rows * 0.6]
    # bottom_right = [cols * 0.9, rows * 0.95]
    # top_right    = [cols * 0.6, rows * 0.6]
    bottom_left  = [cols*0.1, rows*0.95]
    top_left     = [cols*0.4, rows*0.6]
    bottom_right = [cols*0.9, rows*0.95]
    top_right    = [cols*0.6, rows*0.6] 

    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Mapping lines using haugh transform
def hough_transform(image):
    rho = 1
    theta = np.pi/180
    threshold = 30
    minLineLength = 50
    maxLineGap = 300
    return cv2.HoughLinesP(image, rho = rho, theta = theta, threshold = threshold,
                           minLineLength = minLineLength, maxLineGap = maxLineGap)

def map_coordinates(frame, parameters):
    
    height, width, _ = frame.shape
    slope, intercept = parameters
    
    if slope == 0:
        slope = 0.1
    
    y1 = height
    y2 = int(height*0.6)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

# Otimizing the line by joining and streaching them
def optimize_lines(frame, lines):
    height, width, _ = frame.shape
    lane_lines = []
    
    if lines is not None:
        left_fit = []
        right_fit = []        
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)

            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            
            if slope < 0:
                left_fit.append((slope, intercept))
            else:   
                right_fit.append((slope, intercept))

        if len(left_fit) > 0:
            left_fit_average = np.average(left_fit, axis=0)
            lane_lines.append(map_coordinates(frame, left_fit_average))
            
        if len(right_fit) > 0:
            right_fit_average = np.average(right_fit, axis=0)
            lane_lines.append(map_coordinates(frame, right_fit_average))
        
    return lane_lines

# Drawing lines on the image
def draw_lines(image, lines, color = [255, 0, 0], thickness = 2):
    image = np.copy(image)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

# Changing the prespective of the region of interst got get the top view
def warp_perspective(frame):
    height, width = frame.shape[:2]
    # (540, 960)

    source_points = np.float32(
                      [
                        [(425 / 960) * width, (33/54) * height],
                        [(112.5 / 960) * width, height],
                        [(900 / 960) * width, height],
                        [(565 / 960) * width, (33/54) * height],
                    ]
                  )
    offset = 50
    destination_points = np.float32([[offset, 0],
                      [offset, height],
                      [width-2*offset, height],
                      [width-2*offset, 0]])
    
    matrix = cv2.getPerspectiveTransform(source_points, destination_points) 
    
    skyview = cv2.warpPerspective(frame, matrix, (width, height))    

    return skyview

# Generating the histogram to measure the lane lines
def histogram(frame):
    
    histogram = np.sum(frame, axis=0)   
    midpoint = np.int(histogram.shape[0]/2)    
    
    left_x_base = np.argmax(histogram[:midpoint])
    
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint
    
    return left_x_base, right_x_base

# Center of lines on the image
def get_floating_center(frame, lane_lines):
    
    height, width, _ = frame.shape
    
    if len(lane_lines) == 2:
        left_x1, _, left_x2, _ = lane_lines[0][0]
        right_x1, _, right_x2, _ = lane_lines[1][0]
        
        low_mid = (right_x1 + left_x1) / 2
        up_mid = (right_x2 + left_x2) / 2

    else:
        up_mid = int(width*1.9)
        low_mid = int(width*1.9)
    
    return up_mid, low_mid

# Getting the deviation by comparing the center point of actual line and the detected lines in the projection (based on the distribution of white lines)
def add_text(frame, image_center, left_x_base, right_x_base):

    lane_center = left_x_base + (right_x_base - left_x_base) / 2
    
    deviation = image_center - lane_center

    if deviation >= 108:
        text = "Straight"
    elif deviation <= 108 and deviation >= 94:
        text = "Left"
    elif deviation <= 94:
        text = "Right"
    # print(deviation, image_center, lane_center, text)
    
    cv2.putText(frame, "Final: " + text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return frame



def frame_processing(image):
    # global frame_no
    # print(frame_no)
    color_select = HSL_color_selection(image)
    gray         = gray_scale(color_select)
    smooth       = gaussian_smoothing(gray)
    edges        = canny_detector(smooth)
    masked_image       = region_selection(edges)
    # list_images([masked_image])
    hough_line = hough_transform(masked_image)
    opt_line = optimize_lines(image, hough_line)
    line_image = draw_lines(image, opt_line)
    im = warp_perspective(masked_image)
    # list_images([im])
    left_x_base, right_x_base = histogram(im)
    _, low_center = get_floating_center(image, opt_line)
    final_frame = add_text(line_image, low_center, left_x_base, right_x_base)

    return final_frame

    # im = cv2.resize(im, (final_frame.shape[1], final_frame.shape[0]))

    # return np.concatenate((cv2.merge([im, im, im]), final_frame), axis=1)