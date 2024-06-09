import cv2
import numpy as np
import random
import math
import os
import glob 


# The original images are very large, so I wrote a helper function to conveniently display the reduced copy.
def show_copy(image, name = f'image_copy{random.randint(0, 500)}'):
    image_copy = image.copy()
    image_copy = cv2.resize(image_copy, (800, 600))
    cv2.imshow(name, image_copy)
    cv2.waitKey(0)


def find_dartboard_contours(image, show_steps = False):
    
    # Find the largest contour on the yellow maskV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    contours_yellow = cv2.findContours(mask_yellow, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    contours_yellow = sorted(contours_yellow, key=cv2.contourArea, reverse=True)
    largest_contour = contours_yellow[0]

    # Draw a circle around the largest contour
    center, radius = cv2.minEnclosingCircle(largest_contour)
    x = int(center[0])
    y = int(center[1])
    r = int(radius)

    dartboard_contours = []

    dartboard_contours.append((x, y, r))

    gray = np.zeros(image.shape[:2], 'uint8')
    cv2.drawContours(gray, contours_yellow[:10], -1, (255, 255, 255), 2) 

    if show_steps:
        show_copy(gray, 'yellow_contours')
        cv2.waitKey(0)
    
    cv2.circle(image, (x, y), r, (255, 0, 255), 4) # Draw largest circle on an image

    if show_steps:
        show_copy(image, '1_circle')
        cv2.waitKey(0)

    # We find the next 7 circles by cropping the resulting image so that the already found circles 
    # don't interfere with the algorithm's ability to correctly identify circles
     
    radius_delta = r // 8.75 # Every next max radius should be so much smaller than previous

    blank = np.zeros_like(gray)
    mask = cv2.circle(blank, (x, y), int(r-r/9), 255, -1) # Crop the image to exclude the circle found in this step
    gray = cv2.bitwise_and(gray, gray, mask=mask)

    if show_steps:
        show_copy(gray, '1_cropp')
        cv2.waitKey(0)

    for i in range(7):

        # To detect the smallest circle, we only need to decrease the max radius by half the radius_delta
        if i == 6: 
            # All the params values for cv2.HoughCircles were found by experimenting
            circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=2,
            param1=100,
            param2=25,
            minRadius=1,
            maxRadius=int(dartboard_contours[-1][2] - radius_delta/2)
            )

            circles = np.round(circles[0, :]).astype("int")
            largest_circle_index = np.argmax(circles[:, 2]) # Find the index of the circle with the maximum radius
            (x, y, r) = circles[largest_circle_index] # Get the coordinates and radius of the largest circle
    
            cv2.circle(image, (x, y), r, (255, 0, 255), 4) # Draw largest circle on an image
            
            if show_steps:
                show_copy(image, '8_circle')
                cv2.waitKey(0)

            dartboard_contours.append((x, y, r))


        # To detect all but the largest and smallest circles, we decrease the max radius by radius_delta on every step
        else:
            # All the params values for cv2.HoughCircles were found by experimenting
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=2,
                param1=100,
                param2=41,
                minRadius=1,
                maxRadius=int(dartboard_contours[-1][2] - radius_delta)
            )

            circles = np.round(circles[0, :]).astype("int")
            largest_circle_index = np.argmax(circles[:, 2]) # Find the index of the circle with the maximum radius
            (x, y, r) = circles[largest_circle_index] # Get the coordinates and radius of the largest circle
    
            cv2.circle(image, (x, y), r, (255, 0, 255), 4) # Draw largest circle on an image

            if show_steps:
                show_copy(image, f'{i+2}_circle')
                cv2.waitKey(0)

            dartboard_contours.append((x, y, r))
            
            
            blank = np.zeros_like(gray)
            mask = cv2.circle(blank, (x, y), int(r-r/10), 255, -1) # Crop the image to exclude the circle found in this step
            gray = cv2.bitwise_and(gray, gray, mask=mask)
            
            if show_steps:
                show_copy(gray, f'{i+2}_cropp')
                cv2.waitKey(0)


    return dartboard_contours, image


def get_darts_contours(image):

    # Convert the image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Identify the color ranges for the darts
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    lower_green = np.array([50, 50, 50])
    upper_green = np.array([70, 255, 255])

    # Create masks for the darts
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Visualize masks
    if False:
        show_copy(red_mask, 'Red mask')
        show_copy(green_mask, 'Green mask')

    # Find darts contours
    contours_red = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours_green = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Sort the darts in descending order of contour area. 
    # Select the three largest ones, because there are always three darts of the same color in the game
    contours_red = sorted(contours_red, key=cv2.contourArea, reverse=True)
    contours_red = contours_red[:3]

    contours_green = sorted(contours_green, key=cv2.contourArea, reverse=True)
    contours_green = contours_green[:3]



    # If the area of the largest contour is more than 10 times the area of the smallest one, 
    # then the contours of two darts of the same color have merged and we need to separate them.
    if cv2.contourArea(contours_red[0]) > cv2.contourArea(contours_red[2]) * 10:
        masked_red = cv2.bitwise_and(image, image, mask= red_mask)
        contours_red = separate_close_contours(masked_red)

    if cv2.contourArea(contours_green[0]) > cv2.contourArea(contours_green[2]) * 10:
        masked_green = cv2.bitwise_and(image, image, mask= green_mask)
        contours_green = separate_close_contours(masked_green)

    return contours_red, contours_green


def separate_close_contours(masked_image, kernel=(15,15), num_contours=3): 

    # Convert the image to grayscale colors
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold value for dart allocation
    _, thresholded = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

    # Create a kernel for morphological operations
    kernel = np.ones(kernel, np.uint8)  # The value (15,15) for dart separation was found by experimentation

    # Apply erosion and dilation on binary image to separate objects
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

    # Find the contours in the opened image
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the darts in descending order of contour area. 
    # Select the three largest ones, because there are always three darts of the same color in the game
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = contours[:num_contours]

    return contours
    

def get_heads_contours(image):

    # The range of head colors I'm looking for is close to the yellow range. 
    # To avoid finding large yellow contours that are behind the target, I first cut the target out of the image.
    cropped_dartboard = get_cropped_dartboard(image)

    # Visualize cropped dartboard
    if False:
        show_copy(cropped_dartboard, 'Cropped Dartboard')

    # Convert the image to HSV
    hsv = cv2.cvtColor(cropped_dartboard, cv2.COLOR_BGR2HSV)

    # Identify the ranges of brown
    lower_braun = np.array([10, 5, 5])  # This values were found by experimentations
    upper_braun = np.array([19, 150, 150])  

    # Create masks for the heads
    braun_mask = cv2.inRange(hsv, lower_braun, upper_braun)

    if False:
        show_copy(braun_mask, 'Braun Mask')

    # Decrease noise and binarize
    braun_mask = cv2.GaussianBlur(braun_mask, (3,3), 0)
    _, thresholded = cv2.threshold(braun_mask, 30, 255, cv2.THRESH_BINARY)

    # Find and sort the contours by decreasing area, and then select the 6 largest, 
    # because that's how many heads we're looking for
    contours_braun = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours_braun = sorted(contours_braun, key=cv2.contourArea, reverse=True)
    contours_braun = contours_braun[:6]

    # Some of heads contours can merge, so we should use separate_close_contours function again
    blank = np.zeros(image.shape[:2], dtype = 'uint8')
    mask = cv2.drawContours(blank, contours_braun, -1, 255, -1)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # The kernel value was found by experimentations
    separated_contours = separate_close_contours(masked_image, kernel = (3, 3), num_contours = 6)

    return separated_contours


def get_cropped_dartboard(image):

    # Find the largest contour on the yellow maskV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    contours_yellow = cv2.findContours(mask_yellow, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    contours_yellow = sorted(contours_yellow, key=cv2.contourArea, reverse=True)
    largest_contour = contours_yellow[0]

    # Visualize the largest contour
    if False:
        cv2.drawContours(image, [largest_contour], -1, (255,0,0), 10)
        show_copy(image, 'Largest Contour')

    # Draw a circle around the largest contour
    center, radius = cv2.minEnclosingCircle(largest_contour)

    if False:
        cv2.circle(image, (int(center[0]), int(center[1])), int(radius), (255, 255, 255), 10)
        show_copy(image, 'circle')


    # Cropp everything inside the largest contour
    blank = np.zeros_like(image)

    mask = cv2.circle(blank, (int(center[0]), int(center[1])), int(radius), (255, 255, 255), -1)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    cropped_dartboard = cv2.bitwise_and(image, image, mask = mask)

    #show_copy(masked)

    return cropped_dartboard


def find_darts_coordinates(darts_contours, heads_contours, image_shape, mode = 'middle'):
    
    # The mode parameter determines how the location of a dart that failed to be assigned 
    # a head will be determined. This parameter can be set to 'middle' (by default) or 'top'.
    # If mode = 'middle', the location of the headless dart will be equal to the middle pixel of its contour.
    # If mode = 'top', the location of the headless dart will be equal to the top pixel of its contour.

    # Find darts masks and heads masks
    blank = np.zeros(image_shape[:2], 'uint8')
    darts_masks = []

    for dart_contour in darts_contours:
        blank_copy = blank.copy()
        dart_mask = cv2.drawContours(blank_copy, [dart_contour], -1, 255, -1)
        darts_masks.append(dart_mask)
        
    heads_masks = []

    for head_contour in heads_contours:
        blank_copy = blank.copy()
        head_mask = cv2.drawContours(blank_copy, [head_contour], -1, 255, -1)
        heads_masks.append(head_mask)

    darts_coordinates = [] 

    # Let's find the intersection of the darts masks and the heads masks. 
    # The head with the largest intersection will be associated with the dart.

    for i, dart_mask in enumerate(darts_masks):

        intersections = [] # Intersections of dart and every head in pixels

        for head_mask in heads_masks:
            
            kernel = np.ones((5, 5), np.uint8)

            # Magnify the masks to get the intersection
            dilated_dart_mask = cv2.dilate(dart_mask, kernel, iterations = 3)
            dilated_head_mask = cv2.dilate(head_mask, kernel, iterations = 3)

            # Count number of common pixels
            intersection = cv2.bitwise_and(dilated_dart_mask, dilated_head_mask)
            common_pixels = cv2.countNonZero(intersection)
            intersections.append(common_pixels)
        
        max_intersection_index = np.argmax(intersections) # index of biggest intersection


        # If the dart intersects with the head, 
        # the coordinates of the dart will be equal to the center of the head
        if intersections[max_intersection_index] != 0:

            M = cv2.moments(heads_contours[max_intersection_index])
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])

            centroid = (cX, cY)
            darts_coordinates.append(centroid)

        # If the dart contour does not intersect with any head, 
        # the dart coordinates will be equal to the topmost pixel of the contour
        else:
            
            if mode == 'middle':
                M = cv2.moments(darts_contours[i])
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])

                centroid = (cX, cY)
                darts_coordinates.append(centroid)

            if mode == 'top':
                topmost = tuple(darts_contours[i][darts_contours[i][:, :, 1].argmin()][0])
                darts_coordinates.append(topmost)

    return darts_coordinates


def darts_coordinates_to_score(dartboard_contours, darts_coordinates):
    
    x_center = 0
    y_center = 0

    radiuses = []

    for circle in dartboard_contours:
        
        # Multiply the center coordinates and radiuses by 5.8, because the dartboard contourscles 
        # were found using the Hough algorithm on an image reduced by a factor of 5.8.

        x_center += circle[0] * 5.8 
        y_center += circle[1] * 5.8

        # Store all radiuses in one variable 
        radiuses.insert(0, int(circle[2] * 5.8))

    # Find dartboard center as average of 8 circles centers
    x_center //= 8
    y_center //= 8

    field_center = (x_center, y_center)

    # Sort the radiuses and corresponding scores from the inner circle to the outer circle.
    radiuses = sorted(radiuses)
    zones_scores = [100, 80, 60, 50, 40, 30, 20, 10]

    darts_scores = []

    for point in darts_coordinates:

        # For every dat find the distance to center
        distance_to_center = int(math.sqrt((point[0] - field_center[0])**2 + (point[1] - field_center[1])**2))

        # If distanse to center is bigger than the largest radius, score will be equal to 0
        if distance_to_center > radiuses[-1]:
            darts_scores.append(0)

        else:
            # Assign points to the dart that corresponds to the zone it hit
            for i, radius in enumerate(radiuses):

                if distance_to_center <= radius:
                    darts_scores.append(zones_scores[i])
                    break
    
    return darts_scores


def visualize_score(red_darts_coordinates, green_darts_coordinates, red_score, green_score, image):

    # Write the number of points for each red dart
    for i, point in enumerate(red_darts_coordinates):
        cv2.putText(image, str(red_score[i]), (point[0]-60, point[1]-30), cv2.FONT_HERSHEY_COMPLEX, 
                    3, (0,0,255), 10)
    
    # Write the number of points for each green dart
    for i, point in enumerate(green_darts_coordinates):
        cv2.putText(image, str(green_score[i]), (point[0]-60, point[1]-30), cv2.FONT_HERSHEY_COMPLEX, 
                    3, (0,255,0), 10)
        
    
    # Draw every green and red point
    for point in red_darts_coordinates:
        cv2.circle(image, point, 20, (0,0,255), -1)

    for point in green_darts_coordinates:
        cv2.circle(image, point, 20, (0,255, 0), -1)

    # Create a text that contains the result of the game
    first_summary_str = f'RED SCORE: {sum(red_score)}'
    second_summary_str = f'GREEN SCORE: {sum(green_score)}'

    if sum(red_score) > sum(green_score):
        third_summary_string = f'WINNER: RED PLAYER'
    elif sum(red_score) == sum(green_score):
        third_summary_string = f'RESULT: DRAW'
    else:
        third_summary_string = f'WINNER: GREEN PLAYER'


    # Write a text that contains the result of the game
    cv2.putText(image, first_summary_str, (100, 200), cv2.FONT_HERSHEY_COMPLEX, 
                    4, (0,0,200), 15)
    
    cv2.putText(image, second_summary_str, (100, 350), cv2.FONT_HERSHEY_COMPLEX, 
                    4, (0,200,0), 15)
    
    cv2.putText(image, third_summary_string, (100, 500), cv2.FONT_HERSHEY_COMPLEX, 
                    4, (255,255,255), 25)

    cv2.putText(image, third_summary_string, (100, 500), cv2.FONT_HERSHEY_COMPLEX, 
                    4, (0,0,0), 15)
    
    return image
    

def save_results(path_to_data = 'data', path_to_save_results = 'results'):

    # Get all paths to images and their names
    images_paths = glob.glob(os.path.join(path_to_data, "*"))
    images_names = os.listdir(path_to_data)

    # Create directory to save results if it doesn't exist
    if not os.path.exists(path_to_save_results):
        os.makedirs(path_to_save_results)


    # For every image make a prediction and save it
    for i, image_path in enumerate(images_paths):

        image = cv2.imread(image_path)

        # Find the contours of the dartboard circles
        reduced_image = cv2.resize(image, (int(image.shape[1]/5.8), int(image.shape[0]/5.8))) 
        dartboard_contours, _ = find_dartboard_contours(reduced_image)

        # Find darts contours
        red_darts_contours, green_darts_contours = get_darts_contours(image)

        # Find contours of darts heads
        heads_contours = get_heads_contours(image)

        # Assing every dart to a particular point
        red_darts_coordinates = find_darts_coordinates(red_darts_contours, heads_contours, image.shape)
        green_darts_coordinates = find_darts_coordinates(green_darts_contours, heads_contours, image.shape)

        # Analyze the position of each dart and assign a score
        red_scores = darts_coordinates_to_score(dartboard_contours, red_darts_coordinates)
        green_scores = darts_coordinates_to_score(dartboard_contours, green_darts_coordinates)

        # Visualize the score of each dart and the result of the game
        final_image =visualize_score(red_darts_coordinates, green_darts_coordinates, 
                                     red_scores, green_scores, image)
        
        # Save image
        file_path = os.path.join(path_to_save_results, images_names[i])

        # Save file only if there is no  file with such name in results directory
        if not os.path.exists(file_path):
            cv2.imwrite(file_path, final_image)
    

    



