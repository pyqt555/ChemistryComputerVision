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
def count_objects(frame, prev_frame, count_left, count_right):
    # Compute absolute difference between frames
    frame_diff = np.abs(prev_frame - frame)

    # Apply thresholding to obtain binary image
    _, thresh = cv2.threshold(frame_diff.astype(np.uint8), 30, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
    for contour in contours:
        # Compute the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate the center of the bounding box
        center_x = x + w // 2

        # Check the direction and update counts
        if center_x < frame.shape[1] // 2:
            count_left += 1
        else:
            count_right += 1

        # Draw the bounding box on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return count_left, count_right

# Open video capture
cap = cv2.VideoCapture('vid_cam_02.mp4')  # Replace 'your_video.mp4' with the path to your video file

# Read the first frame
ret, prev_frame = cap.read()
prev_frame = prev_frame[:, 1000:-1]
prev_frame = cv2.rotate(prev_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

last = prev_frame.copy()
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
count_left = 0
count_right = 0

while True:
    try:
        # Read the current frame
        ret, frame = cap.read()
        frame = frame[:, 1000:-1]
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        b_arr = (cv2.GaussianBlur(frame, (bisze, bisze), bstd) - cv2.GaussianBlur(last, (bisze, bisze), bstd)) ** 2 ** 1 / 2
        b_arr = high_cont(b_arr)

        # Convert b_arr to binary image
        _, b_arr = cv2.threshold(b_arr.astype(np.uint8), 30, 255, cv2.THRESH_BINARY)

        # Count objects and update counts
        count_left, count_right = count_objects(b_arr, prev_frame, count_left, count_right)

        # Display the frame
        cv2.imshow('Object Counting', b_arr)

        # Update the previous frame
        prev_frame = b_arr.copy()

        # Break the loop if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    except:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Display the counts
print("Objects moving from left to right:", count_left)
print("Objects moving from right to left:", count_right)