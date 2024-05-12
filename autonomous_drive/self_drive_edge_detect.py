from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
from numpy import RankWarning
import warnings
import cv2
from datetime import datetime
import os

# Function to get an image from the camera and convert it into a usable format
def get_image(camera):
    raw_image = camera.getImage()  
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

# Function to convert a color image to grayscale using OpenCV
def greyscale_cv2(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

# Function to apply the Canny edge detection algorithm to find edges in the grayscale image
def canny_edge(gray_img):
    threshold_1 = 100
    threshold_2 = 250
    canny_img = cv2.Canny(gray_img, threshold_1, threshold_2)
    return canny_img

# Function to define and isolate a region of interest within the image for focused processing
def region_of_interest(gray_img):
    height, width = gray_img.shape
    points = np.array([[(int(1*width/10),height),(int(3.5*width/10),6.5*(height)/10),(int(6.5*width/10),6.5*(height)/10),(int(9*width/10),height)]],dtype=np.int32)
    blank = np.zeros_like(gray_img)
    mask = cv2.fillPoly(blank, points, 255)
    masked_image = cv2.bitwise_and(gray_img, mask)
    return masked_image

# Function to detect lines in the image using the Hough Transform method
def hough_transform(canny_img):
    rho = 1
    theta = np.pi/180
    threshold = 16
    min_line_len = 9
    max_line_gap = 20
    lines = cv2.HoughLinesP(canny_img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

# Function to display lines on an image for visual verification
def show_lines(image, lines):
    line_image = np.zeros_like(image, dtype=np.uint8)
    if lines is not None:
        for line in lines:
            try:
                x1, y1, x2, y2 = map(int, line)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
            except Exception as e:
                print(f"Failed to draw line with {line}: {e}")
    return line_image

# Function to compute the endpoints of a line based on its slope and intercept
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope) if slope != 0 else 0
    x2 = int((y2 - intercept) / slope) if slope != 0 else 0
    x1, y1, x2, y2 = map(lambda v: max(0, min(v, max(image.shape[1]-1, image.shape[0]-1))), [x1, y1, x2, y2])
    return np.array([x1, y1, x2, y2])

# Function to average the slopes and intercepts of detected lines and calculate a representative line
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None 
    for line in lines:
        x1, y1, x2, y2 = line[0]
        try:
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope, intercept = parameters
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        except TypeError:
            continue
    left_fit_average = np.average(left_fit, axis=0) if left_fit else (0, image.shape[0] / 2)
    right_fit_average = np.average(right_fit, axis=0) if right_fit else (0, image.shape[0] / 2)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line], dtype=int), (left_fit_average, right_fit_average)

# Function to display an image on a display
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        print("NONNNNNEEE")
        return None 
    for line in lines:
        x1, y1, x2, y2 = line[0]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RankWarning)
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope, intercept = parameters
        except TypeError:
            continue  # Skip this line if polyfit fails

        if slope < 0:  # Consider fine-tuning this condition for better accuracy
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0) if left_fit else (0, image.shape[0] / 2)
    right_fit_average = np.average(right_fit, axis=0) if right_fit else (0, image.shape[0] / 2)
    
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)

    return np.array([left_line, right_line], dtype=int), (left_fit_average,right_fit_average)


#Display image 
def display_image(display, image):
    # Image to display
    image_rgb = np.dstack((image, image,image,))
    # Display image
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)


#initial angle and speed 
manual_steering = 0
steering_angle = 0
angle = 0.0
# speed=30
speed = 30

# set target speed
def set_speed(kmh):
    global speed            #robot.step(50)

#update steering angle
def set_steering_angle(wheel_angle,limit = 1):
    global angle, steering_angle
    # Check limits of steering
    if (wheel_angle - steering_angle) > limit:
        wheel_angle = steering_angle + limit
    if (wheel_angle - steering_angle) < -limit:
        wheel_angle = steering_angle - limit
    steering_angle = wheel_angle
  
    # limit range of the steering angle
    if wheel_angle > limit:
        wheel_angle = limit
    elif wheel_angle < -limit:
        wheel_angle = -limit
    # update steering angle
    angle = wheel_angle

#validate increment of steering angle
def change_steer_angle(inc):
    global manual_steering
    # Apply increment
    new_manual_steering = manual_steering + inc
    # Validate interval 
    if new_manual_steering <= 25.0 and new_manual_steering >= -25.0: 
        manual_steering = new_manual_steering
        set_steering_angle(manual_steering * 0.07)
    # Debugging
    if manual_steering == 0:
        print("going straight")
    else:
        turn = "left" if steering_angle < 0 else "right"
        print("turning {} rad {}".format(str(steering_angle),turn))



# Function to Adjust Car Steering Based on Detected Line Slopes
# This function translates the slopes of detected lines into steering adjustments to control the car.
# The 'steering_sensitivity' parameter amplifies or moderates the effect of slope on steering adjustments.
def adjust_steering_based_on_slope(m_left, m_right,steering_sensitivity = 1):
    # Calculate desired steering change
    steering_change = 0
    # Determine the steering change based on the slopes:
    # If the left slope (m_left) is negative, it suggests the need to steer right to correct the path.
    # Conversely, a positive right slope (m_right) suggests the need to steer left.
    if m_left < 0:
        steering_change = -m_right * steering_sensitivity
    elif m_right > 0:
        steering_change = -m_left * steering_sensitivity
    
    # Apply the steering change
    change_steer_angle(steering_change)

    # Logging for debugging
    print(f"Adjusting steering by {steering_change}: Left Slope = {m_left}, Right Slope = {m_right}")

def detect_intersection(lines):
    # Simple example of intersection detection based on the number of lines.
    # This is quite simplified and should be adjusted according to the specific characteristics of your environment.
    return len(lines) > 4  # Assumes that an intersection has more than 2 lines

def turn_right():
    # Function to adjust the steering angle to turn right
    global steering_angle
    # Set the desired steering angle to turn right, adjust as necessary
    desired_angle = -1  # Assumes that -0.5 is sufficient to turn right, adjust according to your car's sensitivity
    set_steering_angle(0.119)  # Adjust to the desired angle from the current angle

# main
def main():
    # Create the Robot instance.
    robot = Car()
    driver = Driver()

    # Get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # Create camera instance
    camera = robot.getDevice("camera")
    camera.enable(timestep)  # timestep

    # processing display
    display_img = Display("display_image")

    #create keyboard instance
    keyboard=Keyboard()
    keyboard.enable(timestep)
    counter =0

    while robot.step() != -1:
        # Get image from camera
        image = get_image(camera)

        # Process and display image 
        grey_image = greyscale_cv2(image)

        # PROCESS IMAGE

        # 1) Canny image
        canny_image = canny_edge(grey_image)

        # 2) Region of Interest
        rio_image = region_of_interest(canny_image)


        # # 3) Hough Lines
        hough_lines = hough_transform(rio_image)
        # display_image(display_img, rio_image)
        if(counter ==0):
            if hough_lines is not None and len(hough_lines) > 0:
            # # 4) Average Line
                if detect_intersection(hough_lines):
                    print("DETECTED INTERSECTION!")
                    
                    turn_right()  # Gira a la derecha en la intersección
                    set_speed(15)  # Reduce la velocidad durante el giro
                    counter =500
                else:
                    # Lógica normal para seguir líneas
                    avg_lines, fits = average_slope_intercept(grey_image, hough_lines)
                    image_w_lines = show_lines(grey_image, avg_lines)
                    image_with_lines = cv2.addWeighted(rio_image, 1, image_w_lines, 1, 0)
                    display_image(display_img, image_with_lines)
                    if fits:
                        m_left, l_inter = fits[0]
                        m_right, r_inter = fits[1]
                        adjust_steering_based_on_slope(m_left, m_right,2.3)
                        set_speed(30)  # Velocidad normal
            else:
                display_image(display_img, rio_image)
                set_steering_angle(0)
                set_speed(20)
        else:
            counter -=1
            
        #update angle and speed
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)


if __name__ == "__main__":
    main()