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
def count_objects(frame, prev_frame, count_cross, directions, middle_line):
    # Compute absolute difference between frames
    frame_diff = np.abs(prev_frame - frame)

    # Apply thresholding to obtain binary image
    _, thresh = cv2.threshold(frame_diff.astype(np.uint8), 30, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    directions=[]
    # Loop over the contours
    for contour in contours:
        # Compute the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate the center of the bounding box
        center_x = x + w // 2

        # Check if the object crosses the middle line
        if x < middle_line < x + w:
            count_cross += 1
            direction = 1 if center_x < middle_line else -1
            directions.append(direction)

        # Draw the bounding box on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw the middle line
    cv2.line(frame, (middle_line, 0), (middle_line, frame.shape[0]), (0, 0, 255), 2)

    return count_cross, directions

# Open video capture
cap = cv2.VideoCapture('vid_cam_02.mp4')  # Replace 'your_video.mp4' with the path to your video file

# Read the first frame
ret, prev_frame = cap.read()
prev_frame = prev_frame[:, 1000:-1]
prev_frame = cv2.rotate(prev_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

last = prev_frame.copy()
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
count_cross = 0
directions = []

# Set the position of the middle line
middle_line = prev_frame.shape[1] // 2

while True:
    # Read the current frame
    ret, frame = cap.read()
    frame = frame[:, 1000:-1]
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    b_arr = (cv2.GaussianBlur(frame, (bisze, bisze), bstd) - cv2.GaussianBlur(last, (bisze, bisze), bstd)) ** 2 ** 1 / 2
    b_arr = high_cont(b_arr)

    # Convert b_arr to binary image
    _, b_arr = cv2.threshold(b_arr.astype(np.uint8), 30, 255, cv2.THRESH_BINARY)

    # Count objects and update counts
    count_cross, directions = count_objects(b_arr, prev_frame, count_cross, directions, middle_line)

    # Visualize the direction with arrows
    for direction in directions:
        if direction == 1:
            cv2.arrowedLine(frame, (middle_line, frame.shape[0] // 2), (frame.shape[1], frame.shape[0] // 2),
                            (0, 255, 0), 2, tipLength=0.05)
        elif direction == -1:
            cv2.arrowedLine(frame, (middle_line, frame.shape[0] // 2), (0, frame.shape[0] // 2),
                            (0, 255, 0), 2, tipLength=0.05)

    # Display the frame
    cv2.imshow('Object Counting', frame)

    # Update the previous frame
    prev_frame = b_arr.copy()

    # Break the loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Print the counts to the console
print("Objects crossing the middle line:", count_cross)
print("Directions of objects:", directions)