import cv2
import numpy as np

bisze = 31
bstd = 5

def high_cont(mat):
    pre_max = np.max(mat)
    mat = mat.sum(-1) - 50
    mat = mat / np.max(mat)
    mat = cv2.GaussianBlur(mat, (21, 21), 6)
    return mat * pre_max

# Function to count objects and their direction
def count_objects(frame, prev_frame, middle_line, draw_frame, tracked_objects, counters):
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

        # Check if the object is close to the middle line
        if abs(center_x - middle_line) < 50:
            direction = 1 if center_x < middle_line else -1
            objects.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'center_x': center_x,
                'direction': direction,
                'on_line': True if x < middle_line < x + w else False
            })

    # Update tracked objects and draw visualizations
    for obj in tracked_objects:
        # Update the 'on_line' property
        obj['on_line'] = obj['x'] < middle_line < obj['x'] + obj['w']

        # Draw bounding box
        color = (0, 255, 0) if obj['on_line'] else (0, 0, 255)
        cv2.rectangle(draw_frame, (obj['x'], obj['y']), (obj['x'] + obj['w'], obj['y'] + obj['h']), color, 2)

        arrow_length = 30
        arrow_tip = (obj['center_x'] + obj['direction'] * arrow_length, obj['y'] + 10)
        cv2.arrowedLine(draw_frame, (obj['center_x'], obj['y'] + 10), arrow_tip, color, 2)

        # Update counters based on direction
        if obj['on_line']:
            counters[obj['direction']] += 1

    # Draw the middle line
    cv2.line(draw_frame, (middle_line, 0), (middle_line, draw_frame.shape[0]), (0, 0, 255), 2)

    # Display real-time counters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2

    left_counter_text = f"Left: {counters[-1]}"
    right_counter_text = f"Right: {counters[1]}"

    # Get the size of the text
    left_counter_size = cv2.getTextSize(left_counter_text, font, font_scale, font_thickness)[0]
    right_counter_size = cv2.getTextSize(right_counter_text, font, font_scale, font_thickness)[0]

    # Position of the text
    left_counter_position = ((middle_line - left_counter_size[0]) // 2, left_counter_size[1] + 10)
    right_counter_position = (middle_line + (middle_line - right_counter_size[0]) // 2, right_counter_size[1] + 10)

    cv2.putText(draw_frame, left_counter_text, left_counter_position,
                font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    cv2.putText(draw_frame, right_counter_text, right_counter_position,
                font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return objects

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

last = prev_frame.copy()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

# Set the position of the middle line
middle_line = prev_frame.shape[1] // 2

# List to store tracked objects
tracked_objects = []

# Initialize counters for left and right directions
counters = {-1: 0, 1: 0}

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Check if the frame is successfully read
    if not ret:
        print("Error: Failed to read a frame from the video")
        break

    frame = frame[:, 1000:-1]
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    b_arr = (cv2.GaussianBlur(frame, (bisze, bisze), bstd) - cv2.GaussianBlur(last, (bisze, bisze), bstd)) ** 2 ** 1 / 2
    b_arr = high_cont(b_arr)
    last = frame.copy()

    # Count objects and update counts
    tracked_objects = count_objects(b_arr, prev_frame_gray, middle_line, frame, tracked_objects, counters)

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

# Print information about tracked objects to the console
print("Tracked Objects:")
for obj in tracked_objects:
    print(obj)