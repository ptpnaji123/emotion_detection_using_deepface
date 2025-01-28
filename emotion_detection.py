import cv2
from deepface import DeepFace
import time

face_cascade_path = r"D:\csv2\opencv\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

cap = cv2.VideoCapture(r"D:\csv2\opencv\smile2.mov")

# Get video frame rate
fps = int(cap.get(cv2.CAP_PROP_FPS))
delay = int(1000 / fps)

frame_skip = 2  # Process every 2nd frame
frame_count = 0

while True:
    start_time = time.time()  # Start frame timer
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame or video ended.")
        break

    if frame_count % frame_skip != 0:
        frame_count += 1
        continue
    frame_count += 1

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)
    )
    print(f"Faces detected: {len(faces)}")

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (48, 48))  # Resize for faster DeepFace analysis

        # Perform emotion analysis on the face ROI
        try:
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
        except Exception as e:
            emotion = "Error"
            print(f"DeepFace error: {e}")

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display frame
    cv2.imshow('Emotion Detection from Video', frame)
    print(f"Time taken for frame: {time.time() - start_time:.2f} seconds")

    # Synchronize playback speed with video frame rate
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
