
import cv2
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Start capturing video from the webcam (camera index 0)
video = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = video.read()
    if not ret:
        print("Error: Unable to capture frame from the webcam.")
        break

    # Convert BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to get hand landmarks
    results = hands.process(rgb_frame)

    # Draw landmarks on the frame
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        # Extract coordinates of specific landmarks for a belt-like pattern
        x1, y1 = int(landmarks[0].x * frame.shape[1]), int(landmarks[0].y * frame.shape[0])
        x2, y2 = int(landmarks[5].x * frame.shape[1]), int(landmarks[5].y * frame.shape[0])
        x3, y3 = int(landmarks[17].x * frame.shape[1]), int(landmarks[17].y * frame.shape[0])

        # Draw lines between selected landmarks to form a belt-like pattern
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(frame, (x2, y2), (x3, y3), (0, 255, 0), 2)

        # Check if the belt-like pattern is detected
        if y1 < y2 and y2 < y3:
            print("Belt-like pattern detected!")
        else:
            print("No belt-like pattern detected!")
            break  # Break the loop when no belt-like pattern is detected

    # Display the frame
    cv2.imshow("Belt Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object.
video.release()
cv2.destroyAllWindows()

"""
import cv2
import mediapipe as mp
from twilio.rest import Client

# Twilio credentials
TWILIO_SID = "AC96c7febf2fae5654aa081c239cd4a0c1"
TWILIO_AUTH_TOKEN = "6ef76ae348d1f34808a50c9fe1691435"
TWILIO_PHONE_NUMBER = "+18888617564"
YOUR_PHONE_NUMBER = "+17797752514"

client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

video = cv2.VideoCapture(0)

belt_detected = False  # Flag to track belt detection

while True:
    # Read a frame from the webcam
    ret, frame = video.read()
    if not ret:
        print("Error: Unable to capture frame from the webcam.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        # Extract coordinates of specific landmarks for a belt-like pattern
        x1, y1 = int(landmarks[0].x * frame.shape[1]), int(landmarks[0].y * frame.shape[0])
        x2, y2 = int(landmarks[5].x * frame.shape[1]), int(landmarks[5].y * frame.shape[0])
        x3, y3 = int(landmarks[17].x * frame.shape[1]), int(landmarks[17].y * frame.shape[0])

        if y1 < y2 and y2 < y3:
            print("Belt-like pattern detected!")
            belt_detected = True
        else:
            print("No belt-like pattern detected!")
            belt_detected = False

    cv2.imshow("Belt Detection", frame)

    if not belt_detected:
        # Place the phone call once the belt is not detected
        call = client.calls.create(
            to=YOUR_PHONE_NUMBER,
            from_=TWILIO_PHONE_NUMBER,
            url="http://demo.twilio.com/docs/voice.xml"  # Replace with your own TwiML URL
        )
        break  # Exit the loop after placing the call

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
"""