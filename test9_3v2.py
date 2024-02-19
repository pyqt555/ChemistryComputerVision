import cv2
import numpy as np

bisze = 31
bstd = 5
cooldown=5
cd_width=20
def high_cont(mat):
    pre_max = np.max(mat)
    mat = mat.sum(-1) - 50
    mat = mat / np.max(mat)
    mat = cv2.GaussianBlur(mat, (21, 21), 6)
    return mat * pre_max

# Function to count objects and their direction
def count_objects(frame, prev_frame, middle_line, draw_frame,cds):


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
        center_x = x + w // 2
        center_y = y+(h//2)

        # Check if the object is close to the middle line
        if abs(center_x - middle_line) < 50 and cds[center_y]<1:
            direction = 1 if center_x < middle_line else -1
            objects.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'center_x': center_x,
                'direction': direction
            })

            # Draw arrow indicating the direction
            arrow_length = 30
            arrow_tip = (center_x + direction * arrow_length, (y + h) // 2)
            cv2.arrowedLine(draw_frame, (center_x, (y + h) // 2), arrow_tip, (255, 0, 0), 2)
            
        for i in range(2*cd_width):
            #ugly, fix later
            try:
                cds[y-cd_width+i-1]+=5
            except:
                pass
    # Draw the middle line
    #cv2.line(draw_frame, (middle_line, 0), (middle_line, draw_frame.shape[0]), (0, 0, 255), 2)
    for i in range(len(cds)):
        cooldown_color = (0, int(255 - 50 * cds[i]), int(50 * cds[i]))
        draw_frame[i][middle_line] = cooldown_color
    # Draw bounding boxes for detected objects
    for obj in objects:
        cv2.rectangle(draw_frame, (obj['x'], obj['y']), (obj['x'] + obj['w'], obj['y'] + obj['h']), (0, 255, 0), 2)

    return len(objects), objects

# Open video capture
video_path = 'vid_cam_02.mp4'  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

# Check if the video file is successfully opened
if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'")
    exit()

# Read the first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error: Failed to read the first frame from the video")
    exit()

prev_frame = prev_frame[:, 1000:-1]
prev_frame = cv2.rotate(prev_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

# Set the position of the middle line
middle_line = prev_frame.shape[1] // 2
cooldowns=np.zeros(prev_frame.shape[0])




while True:
    # Read the current frame
    ret, frame = cap.read()

    # Check if the frame is successfully read
    if not ret:
        print("Error: Failed to read a frame from the video")
        break

    frame = frame[:, 1000:-1]
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    b_arr = (cv2.GaussianBlur(frame, (bisze, bisze), bstd) - cv2.GaussianBlur(prev_frame, (bisze, bisze), bstd)) ** 2 ** 1 / 2
    b_arr = high_cont(b_arr)

    # Count objects and update counts
    count, objects = count_objects(b_arr, prev_frame_gray, middle_line, frame,cooldowns)

    # Display the frame
    cv2.imshow('Object Counting', frame)

    # Update the previous frame
    prev_frame_gray = b_arr.copy()

    # Break the loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Print the counts and information about objects to the console
print("Number of objects:", count)
for obj in objects:
    print("Object:", obj)