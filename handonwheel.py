from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
from twilio.rest import Client
import torch
import torchvision.transforms as transforms
import torchvision.models.detection as models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from concurrent.futures import ThreadPoolExecutor

# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

ap = argparse.ArgumentParser()
ap.add_argument("--phone-number", required=True, help="your phone number in E.164 format, e.g., +1234567890")
args = vars(ap.parse_args())

TWILIO_SID = "AC96c7febf2fae5654aa081c239cd4a0c1"
TWILIO_AUTH_TOKEN = "6ef76ae348d1f34808a50c9fe1691435"
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

def draw_hands(frame, hands):
    for hand in hands:
        x_min, y_min, x_max, y_max = hand
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        cv2.line(frame, (center_x, center_y), (x_min, y_min), (0, 255, 0), 2)
        cv2.line(frame, (center_x, center_y), (x_max, y_min), (0, 255, 0), 2)

def preprocess_frame_for_detection(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(frame).unsqueeze(0)

def detect_hands(frame, roi):
    hand_detection_model = fasterrcnn_resnet50_fpn(pretrained=True)
    hand_detection_model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    preprocessed_roi = transform(roi).unsqueeze(0)

    with torch.no_grad():
        hand_detections = hand_detection_model(preprocessed_roi)

    hand_boxes = hand_detections[0]['boxes']
    hand_scores = hand_detections[0]['scores']

    confident_hands = hand_boxes[hand_scores > 0.5]

    # Print the detected hands for debugging
    print("Detected Hands:", confident_hands)

    return confident_hands

def hands_in_expected_position(hands):
    if len(hands) != 2:
        return False

    y_coords = [hand[1] for hand in hands]
    raised_condition = y_coords[0] < y_coords[1]
    lowered_condition = y_coords[0] > y_coords[1]

    if raised_condition or lowered_condition:
        return True
    else:
        return False

print("[INFO] camera sensor warming up...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Function to process a single frame
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Dummy values for face detection (remove eye detection part)
    rects = [dlib.rectangle(0, 0, frame.shape[1], frame.shape[0])]

    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        roi_frame = frame[y:y+h, x:x+w]
        preprocessed_roi_frame = preprocess_frame_for_detection(roi_frame)

        with torch.no_grad():
            detections = model(preprocessed_roi_frame)

        arm_detections = detections[0]['boxes'][detections[0]['scores'] > 0.5]

        if len(arm_detections) == 0:
            cv2.putText(frame, "ARMS ALERT!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            call = client.calls.create(
                to=args["phone_number"],
                from_="+18888617564",
                url="https://demo.twilio.com/welcome/voice/"
            )

        hands = detect_hands(frame, roi_frame)

        # Draw green dots and lines on hands
        draw_hands(frame, hands)

        if not hands_in_expected_position(hands):
            cv2.putText(frame, "HANDS ALERT!", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            call = client.calls.create(
                to=args["phone_number"],
                from_="+18888617564",
                url="https://demo.twilio.com/welcome/voice/"
            )

    cv2.imshow("Frame", frame)

# Start the video stream and process frames using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=1) as executor:
    while True:
        frame = vs.read()
        if frame is not None:
            # Process the frame in a separate thread
            executor.submit(process_frame, frame)

            # Display the frame
            cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

cv2.destroyAllWindows()
vs.stop()
