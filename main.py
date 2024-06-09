import cv2
import random


from utils import show_copy, get_darts_contours, get_heads_contours, find_darts_coordinates
from utils import find_dartboard_contours, darts_coordinates_to_score, visualize_score, save_results


# Read an image
image_path = 'data/darts1.jpg'
image = cv2.imread(image_path)

# Find the contours of the dartboard circles
# Shrink the original image to make Hough's algorithm perform faster and more accurately
reduced_image = cv2.resize(image, (int(image.shape[1]/5.8), int(image.shape[0]/5.8))) 

# Set show_steps = True, if you want to see all the steps
dartboard_contours, dartboard = find_dartboard_contours(reduced_image, show_steps = False) 

# Visualize found dartboard contours
if False:
    cv2.imshow('Dartboard Contours', dartboard)


# Find darts contours
red_darts_contours, green_darts_contours = get_darts_contours(image)

# Visualize found darts contours
if False:
    for red_contour in red_darts_contours:
        random_red = random.randint(50,255) # Draw each contour with a different shade of color
        cv2.drawContours(image, [red_contour], -1, (0,0,random_red), 10)

    for green_contour in green_darts_contours:
        random_green = random.randint(50,255) # Draw each contour with a different shade of color
        cv2.drawContours(image, [green_contour], -1, (0,random_green,0), 10)
    
    show_copy(image, 'Darts Contours')



# Find contours of darts heads
heads_contours = get_heads_contours(image)

# Visualize found heads contours
if False: 
    for head_contour in heads_contours:
        random_blue = random.randint(50,255) # Draw each contour with a different shade of color
        cv2.drawContours(image, [head_contour], -1, (random_blue,0,0), 10)
    show_copy(image, 'Heads Contours')


# Assing every dart to a particular point
red_darts_coordinates = find_darts_coordinates(red_darts_contours, heads_contours, image.shape)
green_darts_coordinates = find_darts_coordinates(green_darts_contours, heads_contours, image.shape)

# Visualize found darts coordinates
if False: 
    for point in red_darts_coordinates:
        cv2.circle(image, point, 20, (0,0,255), -1)

    for point in green_darts_coordinates:
        cv2.circle(image, point, 20, (0,255, 0), -1)
    
    show_copy(image, 'Darts Points')


# Analyze the position of each dart and assign a score
red_scores = darts_coordinates_to_score(dartboard_contours, red_darts_coordinates)
green_scores = darts_coordinates_to_score(dartboard_contours, green_darts_coordinates)


# Visualize the score of each dart and the result of the game
visualize_score(red_darts_coordinates, green_darts_coordinates, red_scores, green_scores, image)
show_copy(image, 'Game Results')


# Save results
if False:
    save_results()

cv2.waitKey(0)