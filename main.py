
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
bisze = 31
bstd = 5
cooldown=5
cd_width=50
total_objects_left = 0
total_objects_right = 104
def high_cont(mat):
    
    mat = mat.sum(-1) - 20
    mat = mat / np.max(mat)
    mat = cv2.GaussianBlur(mat, (21, 21), 6)
    return mat * 1

# Function to count objects and their direction
def count_objects(mag,x_direction, middle_line, cds,draw_frame):
    global total_objects_left
    global total_objects_right
    mag=mag*10
    #update cds
    cds=cds-1
    for i in range(len(cds)):
        if cds[i]<0:
            cds[i]=0

    # Compute absolute difference between frames
    right=(x_direction>1)*255
    left=(x_direction<-1)*255
    # Apply thresholding to obtain binary image
    _, thresh = cv2.threshold(cv2.GaussianBlur(mag,(5,5),3).astype(np.uint8), 5, 255, cv2.THRESH_BINARY)

    #cv2.imshow("thresh",thresh)
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
                right_sum=np.sum(right[y:y+h,x:x+w])
                left_sum=np.sum(left[y:y+h,x:x+w])
                direction = 1 if right_sum>left_sum else -1
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
                arrow_length = 50
                arrow_tip = (center_x + direction * arrow_length, center_y)
                cv2.arrowedLine(draw_frame, (center_x, center_y), arrow_tip, (255, 0, 0), 2)
            
            for i in range(2*h):
                #ugly, fix later
                try:
                    cds[y-h+i-1]=cooldown
                except:
                    pass

            
    for obj in objects:
        cv2.rectangle(draw_frame, (obj['x'], obj['y']), (obj['x'] + obj['w'], obj['y'] + obj['h']), (0, 255, 0), 2)
    return len(objects), objects ,cds

# Open video capture
video_path = 'final_stuff\C0006_fixed.mp4'  # Replace with the path to your video file
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
ret, f1 = cap.read()
if not ret:
    print("Error: Failed to read the first frame from the video")
    exit()





f1 = f1[:, 100:-100]
f1 = cv2.rotate(f1, cv2.ROTATE_90_COUNTERCLOCKWISE)
f1=cv2.convertScaleAbs( cv2.GaussianBlur(f1, (61, 61), 30), alpha=3, beta=-1)
f1=cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
resize_dim = 500
max_dim = max(f1.shape)
scale = resize_dim/max_dim
f1 = cv2.resize(f1, None, fx=scale, fy=scale)
# Set the position of the middle line
middle_line = f1.shape[1] // 2
cooldowns=np.zeros(f1.shape[0])

listLeft=[]
listRight=[]




def optical_flow(im1,im2):
    
    gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale)
    #print("Number of channels in im1:", im1.shape)
    #print("Number of channels in im2:", gray.shape)
    
    flow = cv2.calcOpticalFlowFarneback(im1, gray, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
    # Compute the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    
    return magnitude, flow[:,:,0] , gray






c=0
try:
    while True:
        c+=1
        if c%100==0:
            print(c)
        # Read the current frame
        ret, frame = cap.read()
        # Check if the frame is successfully read
        
        frame = frame[:, 100:-100]
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imshow("f",frame)
        drawF=frame.copy()
        drawF=cv2.resize(drawF, None, fx=scale, fy=scale)
        frame=cv2.convertScaleAbs( cv2.GaussianBlur(frame, (61, 61), 30), alpha=3, beta=-1)
        cv2.imshow("f2",frame)
        # Count objects and update counts
        magnitude,x_dist,last_f=optical_flow(f1,frame)
        f1=last_f
        count, objects,cooldowns = count_objects(magnitude,x_dist, middle_line,cooldowns,drawF)
        #cv2.imshow("x",x_dist)
        cv2.imshow("magnitude",magnitude)
        #cv2.imshow("col",frame)
        #cv2.imshow("a",drawF)
        listLeft.append(total_objects_left)
        listRight.append(total_objects_right)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
except:
    print("no more frames")       
# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
plt.plot(listLeft)
plt.plot(listRight)
plt.show()
# Print the counts and information about objects to the console
print("Left: ", total_objects_left, "Right: " ,total_objects_right)









