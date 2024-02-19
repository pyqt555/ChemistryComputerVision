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

def count_objects(frame, middle_line, draw_frame, tracked_objects, counters, object_trackers, tracker_type='CSRT'):
    frame_uint8 = frame.astype(np.uint8)  # Convert frame to 8-bit unsigned integer

    # Update the trackers for existing objects
    for obj_id, tracker in object_trackers.items():
        success, bbox = tracker.update(frame_uint8)

        if success:
            x, y, w, h = map(int, bbox)
            center_x = x + w // 2

            direction = 1 if center_x < middle_line else -1

            tracked_objects.append({
                'id': obj_id,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'center_x': center_x,
                'direction': direction,
                'on_line': x < middle_line < x + w,
                'counted': False  # Initialize 'counted' attribute
            })

            color = (0, 255, 0) if tracked_objects[-1]['on_line'] else (0, 0, 255)
            cv2.rectangle(draw_frame, (x, y), (x + w, y + h), color, 2)

            arrow_length = 30
            arrow_tip = (center_x + direction * arrow_length, (y + h) // 2)
            cv2.arrowedLine(draw_frame, (center_x, (y + h) // 2), arrow_tip, color, 2)

    # Find new objects using contour detection
    _, thresh = cv2.threshold(frame_uint8, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
    
        # Extract the bounding box coordinates
        x, y, w, h = cv2.boundingRect(box)
        center_x = x + w // 2

        # Check if the object is already being tracked
        is_tracked = any(np.array([abs(obj['x'] - x) < 10 and abs(obj['y'] - y) < 10 for obj in tracked_objects]))

        if not is_tracked:
            # Initialize a new tracker for the object
            new_obj_id = max([obj['id'] for obj in tracked_objects], default=0) + 1
            new_tracker = cv2.TrackerCSRT_create() if tracker_type == 'CSRT' else cv2.TrackerKCF_create()

            # Correct bounding box initialization
            new_tracker.init(frame_uint8, (x, y, w, h))

            object_trackers[new_obj_id] = new_tracker

    # Update counters based on object movement across the line
    for obj in tracked_objects:
        if obj['on_line'] and not obj['counted']:
            counters[obj['direction']] += 1
            obj['counted'] = True  # Mark the object as counted to avoid duplicate counting

    cv2.line(draw_frame, (middle_line, 0), (middle_line, draw_frame.shape[0]), (0, 0, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2

    left_counter_text = f"Left: {counters[-1]}"
    right_counter_text = f"Right: {counters[1]}"

    left_counter_size = cv2.getTextSize(left_counter_text, font, font_scale, font_thickness)[0]
    right_counter_size = cv2.getTextSize(right_counter_text, font, font_scale, font_thickness)[0]

    left_counter_position = ((middle_line - left_counter_size[0]) // 2, left_counter_size[1] + 10)
    right_counter_position = (middle_line + (middle_line - right_counter_size[0]) // 2, right_counter_size[1] + 10)

    cv2.putText(draw_frame, left_counter_text, left_counter_position,
                font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    cv2.putText(draw_frame, right_counter_text, right_counter_position,
                font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return tracked_objects

# Open video capture
video_path = 'vid_cam_02.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'")
    exit()

ret, prev_frame = cap.read()
if not ret:
    print("Error: Failed to read the first frame from the video")
    exit()

prev_frame = prev_frame[:, 1000:-1]
prev_frame = cv2.rotate(prev_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
last = prev_frame.copy()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

middle_line = prev_frame.shape[1] // 2

counters = {-1: 0, 1: 0}
trackers = {}
tracked_objects = []

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to read a frame from the video")
        break

    frame = frame[:, 1000:-1]
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    b_arr = (cv2.GaussianBlur(frame, (bisze, bisze), bstd) - cv2.GaussianBlur(last, (bisze, bisze), bstd)) ** 2 ** 1 / 2
    b_arr = high_cont(b_arr)
    last = frame.copy()

    tracked_objects = count_objects(b_arr, middle_line, frame, tracked_objects, counters, trackers)

    cv2.imshow('Object Counting', frame)

    prev_frame_gray = b_arr.copy()

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Tracked Objects:")
for obj in tracked_objects:
    print(obj)