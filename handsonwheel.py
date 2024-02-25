#python mediapipetest.py

# Import the necessary modules.
import cv2
import subprocess
from twilio.rest import Client

# Install mediapipe and twilio using pip
subprocess.run(['pip', 'install', 'mediapipe', 'twilio'])

# Now you can import mediapipe and twilio in your script
import mediapipe as mp

# Twilio credentials
TWILIO_SID = "Your Twilio SID"
TWILIO_AUTH_TOKEN = "Your Twilio Auth Token"
TWILIO_PHONE_NUMBER = "Your Twilio Phone Number"
YOUR_PHONE_NUMBER = "Your Phone Number"

client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# Create a HandLandmarks object.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

# Load the input video.
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Function to draw landmarks on the image
def draw_landmarks_on_image(image, landmarks_list):
    annotated_image = image.copy()
    if landmarks_list:
        for landmarks in landmarks_list:
            for idx, landmark in enumerate(landmarks.landmark):
                h, w, c = annotated_image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(annotated_image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return annotated_image 

# Initialize the hands_detected flag
hands_detected = False

# Detect hand landmarks from each frame of the video.
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to get hand landmarks
    results = hands.process(rgb_frame)

    # Process the classification result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(frame, results.multi_hand_landmarks)
    cv2.imshow("Annotated Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    # Check if hands are detected in the frame
    if results.multi_hand_landmarks:
        hands_detected = True
        #print("Hands detected in the frame")
    else:
        hands_detected = False
        print("No hands detected in the frame")
        # Send a phone call alert
        call = client.calls.create(
            to="Your Phone Number",
            from_="Your Twilio Phone Number",
            url="Twilio URL to answer call"
        )
        break  # Exit the loop when no hands are detected

    # Pause or play the video based on hands detection
    if hands_detected:
        # Pause the video
        # Add your code to pause the video here
        pass
    else:
        # Play the video
        # Add your code to play the video here
        pass

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object.
video.release()
cv2.destroyAllWindows()
