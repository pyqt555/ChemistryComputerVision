

import cv2
import numpy as np
import matplotlib.pyplot as plt
bisze = 31
bstd = 5
cooldown=5
cd_width=50
total_objects_left = 13
total_objects_right = 87
def high_cont(mat):
    pre_max = np.max(mat)
    mat = mat.sum(-1) - 20
    mat = mat / np.max(mat)
    mat = cv2.GaussianBlur(mat, (21, 21), 6)
    return mat * pre_max

# Function to count objects and their direction
def count_objects(frame, prev_frame, middle_line, draw_frame,cds):
    global total_objects_left
    global total_objects_right

    #update cds
    cds=cds-1
    for i in range(len(cds)):
        if cds[i]<0:
            cds[i]=0

    # Compute absolute difference between frames
    frame_diff = np.abs(prev_frame - frame)

    # Apply thresholding to obtain binary image
    _, thresh = cv2.threshold(frame_diff.astype(np.uint8), 30, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to store information about detected objects
    objects = []

    # Loop over the contours
    for contour in contours:
        # Compute the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate the center of the bounding box
        center_x = x + (w // 2)
        center_y = y+(h//2)

        # Check if the object is close to the middle line
        if abs(center_x - middle_line) < 50 :
            if cds[center_y]<1:
                direction = 1 if center_x < middle_line else -1
                objects.append({
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'center_x': center_x,
                    'direction': direction
                })
                if direction == 1:
                    total_objects_right += 1
                    total_objects_left-=1
                else:
                    total_objects_left += 1
                    total_objects_right-=1


                # Draw arrow indicating the direction
                arrow_length = 50
                arrow_tip = (center_x + direction * arrow_length, center_y)
                cv2.arrowedLine(draw_frame, (center_x, center_y), arrow_tip, (255, 0, 0), 2)
            
            for i in range(2*h):
                #ugly, fix later
                try:
                    cds[y-h+i-1]=cooldown
                except:
                    pass
    # Draw the middle line
    #cv2.line(draw_frame, (middle_line, 0), (middle_line, draw_frame.shape[0]), (0, 0, 255), 2)
    for i in range(len(cds)):
        cooldown_color = [0, int(255 - 10 * cds[i]), int(10 * cds[i])]
        draw_frame[i][middle_line] = cooldown_color

    #draw counters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    left_text = f"Total Left: {total_objects_left}"
    right_text = f"Total Right: {total_objects_right}"
    cv2.putText(draw_frame, left_text, (10, 30), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    cv2.putText(draw_frame, right_text, (10, 60), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    # Draw bounding boxes for detected objects
    for obj in objects:
        cv2.rectangle(draw_frame, (obj['x'], obj['y']), (obj['x'] + obj['w'], obj['y'] + obj['h']), (0, 255, 0), 2)

    return len(objects), objects ,cds

# Open video capture
video_path = 'socken\C0006.MP4'  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

# Check if the video file is successfully opened
if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'")
    exit()


def increaseContrast (img):
    return cv2.convertScaleAbs( cv2.GaussianBlur(img, (61, 61), 30), alpha=3, beta=-1)
#cutof sta
for i in range(50*3):
    cap.read()
# Read the first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error: Failed to read the first frame from the video")
    exit()

prev_frame = increaseContrast(prev_frame)[:, 200:-500]
prev_frame = cv2.rotate(prev_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

# Set the position of the middle line
middle_line = prev_frame.shape[1] // 2
cooldowns=np.zeros(prev_frame.shape[0])

listLeft=[]
listRight=[]

c=0
while True:
    c+=1
    
    print(c)
    # Read the current frame
    ret, frame = cap.read()
    # Check if the frame is successfully read
    if not ret:
        print("Error: Failed to read a frame from the video")
        break
    frame=increaseContrast(frame)
    frame = frame[:, 200:-500]
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    b_arr = (cv2.GaussianBlur(frame, (bisze, bisze), bstd) - cv2.GaussianBlur(prev_frame, (bisze, bisze), bstd)) ** 2 ** 1 / 2
    b_arr = high_cont(b_arr)

    # Count objects and update counts
    count, objects,cooldowns = count_objects(b_arr, prev_frame_gray, middle_line, frame,cooldowns)

    # Display the frame
    cv2.imshow('Object Counting', frame)
    cv2.imshow("test1",b_arr)
    cv2.imshow("test2",prev_frame_gray)
    # Update the previous frame
    prev_frame_gray = b_arr.copy()
    listLeft.append(total_objects_left)
    listRight.append(total_objects_right)
    
   
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
plt.plot(listLeft)
plt.plot(listRight)
plt.show()
# Print the counts and information about objects to the console
print("Left: ", total_objects_left, "Right: " ,total_objects_right)
