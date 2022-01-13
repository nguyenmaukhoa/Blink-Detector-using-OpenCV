# Import necessary libraries
import cv2
import numpy as np
import time

# Install pygame to import sound effect
try:
    from pygame import mixer
except ModuleNotFoundError:
    mixer = None
    pass

# -------------------------------------------------------------------
# Initialization
# -------------------------------------------------------------------

# Blink counter
BLINK = 0

# Model file path
MODEL_PATH = './model/res10_300x300_ssd_iter_140000.caffemodel'
CONFIG_PATH = './model/deploy.prototxt'
LBF_MODEL = './model/lbfmodel.yaml'

# Create a face detector network instance.
net = cv2.dnn.readNetFromCaffe(CONFIG_PATH, MODEL_PATH)

# Create the landmark detector instance
landmarkDetector = cv2.face.createFacemarkLBF()
landmarkDetector.loadModel(LBF_MODEL)

# Initialize video capture object
cap = cv2.VideoCapture('pexels-ron-lach-7817602.mp4')
# Use this option for camera use
# cap = cv2.VideoCapture(0)

# Write Video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('media/output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 24, (frame_width, frame_height))

state_prev = state_curr = 'open'


# -------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------

def detect_faces(image, detection_threshold=0.7):
    # Convert image to blob
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])

    # Pass the blob to DNN Model
    net.setInput(blob)

    # Get detections
    detections = net.forward()

    # Face list
    faces = []
    img_h = image.shape[0]
    img_w = image.shape[1]

    # Process the detections
    for detection in detections[0][0]:
        if detection[2] >= detection_threshold:
            left = detection[3] * img_w
            top = detection[4] * img_h
            right = detection[5] * img_w
            bottom = detection[6] * img_h

            face_w = right - left
            face_h = bottom - top

            face_roi = (left, top, face_w, face_h)
            faces.append(face_roi)

    return np.array(faces).astype(int)


def get_primary_face(faces, frame_h, frame_w):
    primary_face_index = None
    face_height_max = 0

    for idx in range(len(faces)):
        face = faces[idx]

        # Confirm bounding box of primary face does not exceed frame size
        x1 = face[0]
        y1 = face[1]
        x2 = x1 + face[2]
        y2 = y1 + face[3]

        if x1 > frame_w or y1 > frame_h or x2 > frame_w or y2 > frame_h:
            continue
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            continue
        if face[3] > face_height_max:
            primary_face_index = idx
            # face_height_max = face[3]

    if primary_face_index is not None:
        primary_face = faces[primary_face_index]
    else:
        primary_face = None

    return primary_face


def visualize_eyes(landmarks):
    for i in range(36, 48):
        cv2.circle(frame, tuple(landmarks[i].astype('int')), 2, (0, 255, 0), -1)


def get_eye_aspect_ratio(landmarks):
    # Calculate the Euclidean distances between 2 sets of vertical eye landmarks
    vert_dist_1right = np.linalg.norm(landmarks[41] - landmarks[37])
    vert_dist_2right = np.linalg.norm(landmarks[40] - landmarks[38])
    vert_dist_1left = np.linalg.norm(landmarks[47] - landmarks[43])
    vert_dist_2left = np.linalg.norm(landmarks[46] - landmarks[44])

    # Calculate the Euclidean distances between 2 sets of vertical eye landmarks
    horz_dist_right = np.linalg.norm(landmarks[39] - landmarks[36])
    horz_dist_left = np.linalg.norm(landmarks[45] - landmarks[42])

    # Compute the eye aspect ratio
    EAR_left = (vert_dist_1left + vert_dist_2left) / (2.0 * horz_dist_left)
    EAR_right = (vert_dist_1right + vert_dist_2right) / (2.0 * horz_dist_right)

    ear = (EAR_left + EAR_right)/2
    return ear

def play(file):
    mixer.init()
    sound = mixer.Sound(file)
    sound.play()


# -------------------------------------------------------------------
# Main Program
# -------------------------------------------------------------------

if __name__ == "__main__":
    frame_count = 0
    frame_calib = 30  # Number of frames used for threshold calibration
    sum_ear = 0

    ret, frame = cap.read()
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print('Unable to read frames')
            break

        # Detect face
        faces = detect_faces(frame, detection_threshold=0.9)

        if len(faces) > 0:
            # Use the primary (largest) fae in the frame
            primary_face = get_primary_face(faces, frame_h, frame_w)

            if primary_face is not None:
                cv2.rectangle(frame, primary_face, (0, 255, 0), 3)

                # Detect landmarks
                retval, landmarksList = landmarkDetector.fit(frame, np.expand_dims(primary_face, 0))

                if retval:
                    # Get the landmarks for the primary face
                    landmarks = landmarksList[0][0]

                    # Visualize detections
                    visualize_eyes(landmarks)

                    # Get the eye aspect ratios
                    ear = get_eye_aspect_ratio(landmarks)

                    # Calibrate threshold based on initial frames
                    if frame_count < frame_calib:
                        frame_count += 1
                        sum_ear += ear

                    elif frame_count == frame_calib:
                        frame_count += 1
                        avg_ear = sum_ear / frame_count
                        # Set high threshold to 90% of average EAR
                        HIGHER_TH = 0.9 * avg_ear
                        # Set low threshold to 70% of high threshold
                        LOWER_TH = 0.8 * HIGHER_TH
                        print("Set Ear High: ", HIGHER_TH)
                        print("Set Ear Low: ", LOWER_TH)

                    else:
                        # We get the blink when the eye status switch from "closed" to "open"
                        if ear < LOWER_TH:
                            state_curr = 'closed'
                            print('State_Closed (EAR): ', ear)
                        elif ear > HIGHER_TH:
                            state_curr = 'open'

                        if state_prev == 'closed' and state_curr == 'open':
                            BLINK += 1
                            print('State-Open (EAR): ', ear)
                            print('BLINK DETECTED\n')
                            if mixer:
                                play('click.wav')
                        # Update the previous state
                        state_prev = state_curr

                        cv2.putText(frame, "Blink Counter: {}".format(BLINK), (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.5, (0, 255, 0), 4, cv2.LINE_AA)
                        # Write Video
                        out.write(frame)
                        cv2.imshow('Output', frame)

                        # Wait and Escape
                        k = cv2.waitKey(1)
                        if k == ord('q'):
                            cv2.destroyAllWindows()
                            break
        else:
            print('No Face Detected!')

    cap.release()
