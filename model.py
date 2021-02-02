import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_pose2 = mp.solutions.pose
mp_holistic = mp.solutions.holistic

# For webcam input:
pose = mp_pose.Pose(
    min_detection_confidence=0.25, min_tracking_confidence=0.25)
pose2 = mp_pose2.Pose(
    min_detection_confidence=0.25, min_tracking_confidence=0.25)
cap = cv2.VideoCapture(0)
number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
while cap.isOpened():
  success, image = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    # If loading a video, use 'break' instead of 'continue'.
    continue

  # Flip the image horizontally for a later selfie-view display, and convert
  # the BGR image to RGB.
  image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  image.flags.writeable = False
  results = pose.process(image)
  results2 = pose2.process(image)


  # Draw the pose annotation on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  mp_drawing.draw_landmarks(
      image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
  cv2.imshow('MediaPipe Pose', image)
  print(mp_pose.POSE_CONNECTIONS)

  if cv2.waitKey(5) & 0xFF == 27:
    break
pose.close()
cap.release()

