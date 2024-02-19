import cv2
import numpy as np

# Get a VideoCapture object from video and store it in vs
vc = cv2.VideoCapture("final_stuff\C0006_fixed.mp4")
# Read first frame
ret, first_frame = vc.read()
first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
# Scale and resize image
resize_dim = 1000
max_dim = max(first_frame.shape)
scale = resize_dim/max_dim
first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
# Convert to gray scale 
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)


# Create mask
mask = np.zeros_like(first_frame)
# Sets image saturation to maximum
mask[..., 1] = 255
anglem=np.zeros_like(first_frame)


for i in range(300):
    frame=vc.read()

while(vc.isOpened()):
    # Read a frame from video
    ret, frame = vc.read()
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame=cv2.convertScaleAbs( cv2.GaussianBlur(frame, (41, 41), 20), alpha=3, beta=-1)
    
    # Convert new frame format`s to gray scale and resize gray frame obtained
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale)

    # Calculate dense optical flow by Farneback method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
    # Compute the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Set image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Set image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Convert HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    
    # Resize frame size to match dimensions
    frame = cv2.resize(frame, None, fx=scale, fy=scale)
    #anglem[:,:,1]=((flow[:,:,0]>0)*flow[:,:,0])#positive/negative x(actually its y) direction
    #anglem[:,:,2]=((flow[:,:,0]<0)*flow[:,:,0])
    anglem[:,:,0]=(flow[:,:,0]>1)*255#(((flow[..., 0])/15)*255)*
    anglem[:,:,1]=(flow[:,:,0]<-1)*255#(((flow[..., 0])/15)*255)*
    #anglem[:,:,2]=np.multiply(np.multiply(np.cos(angle),magnitude),np.cos(angle)>0)*10
    #anglem[:,:,1]=np.multiply(np.multiply(np.cos(angle),magnitude),np.cos(angle)<0)*10
    # Open a new window and displays the output frame
    dense_flow = cv2.addWeighted(frame, 1,rgb, 2, -.6)
    cv2.imshow("Dense optical flow", dense_flow)
    cv2.imshow("x",anglem)
    cv2.imshow("d",magnitude)

    
    
    
    # Update previous frame
    prev_gray = gray
    # Frame are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# The following frees up resources and closes all windows
vc.release()

cv2.destroyAllWindows()