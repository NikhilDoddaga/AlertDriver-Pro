import cv2
import numpy as np
from twilio.rest import Client

# Slope of line
def Slope(a, b, c, d):
    return (d - b) / (c - a)

# Twilio credentials
TWILIO_SID = "AC96c7febf2fae5654aa081c239cd4a0c1"
TWILIO_AUTH_TOKEN = "6ef76ae348d1f34808a50c9fe1691435"
TWILIO_PHONE_NUMBER = "+18888617564"
YOUR_PHONE_NUMBER = "+17797752514"

client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# Create a Twilio client
twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# Open a connection to the webcam (assuming it's the first camera, you might need to adjust the index)
cap = cv2.VideoCapture(0)

# Number of frames to wait before displaying "No Belt Detected"
no_belt_frames_threshold = 30
no_belt_frames_count = 0

# Number of frames to display "Belt Detected" after detection (increased duration)
belt_detected_frames_threshold = 60
belt_detected_frames_count = 0

while True:
    # Read a frame from the webcam
    ret, beltframe = cap.read()

    # Resize the frame
    beltframe = cv2.resize(beltframe, (800, 600))

    # Convert the frame to grayscale
    beltgray = cv2.cvtColor(beltframe, cv2.COLOR_BGR2GRAY)

    # Blur the image for smoothness
    blur = cv2.blur(beltgray, (1, 1))

    # Convert image to edges
    edges = cv2.Canny(blur, 50, 400)

    # Previous Line Slope
    ps = 0

    # Previous Line Co-ordinates
    px1, py1, px2, py2 = 0, 0, 0, 0

    # No Belt Detected Yet
    belt_detected = False

    # Extracting Lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/270, 30, maxLineGap=20, minLineLength=170)

    # If "lines" is not empty
    if lines is not None:
        # Loop line by line
        for line in lines:
            # Co-ordinates Of Current Line
            x1, y1, x2, y2 = line[0]

            # Slope Of Current Line
            s = Slope(x1, y1, x2, y2)

            # If Current Line's Slope Is Greater Than 0.7 And Less Than 2
            if 0.7 < abs(s) < 2:
                # And Previous Line's Slope Is Within 0.7 To 2
                if 0.7 < abs(ps) < 2:
                    # And Both The Lines Are Not Too Far From Each Other
                    if (abs(x1 - px1) > 5 and abs(x2 - px2) > 5) or (abs(y1 - py1) > 5 and abs(y2 - py2) > 5):
                        # Plot The Lines On "beltframe"
                        cv2.line(beltframe, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.line(beltframe, (px1, py1), (px2, py2), (0, 0, 255), 3)

                        # Belt Is Detected
                        belt_detected = True
                        belt_detected_frames_count = belt_detected_frames_threshold

            # Otherwise Current Slope Becomes Previous Slope (ps) And Current Line Becomes Previous Line (px1, py1, px2, py2)
            ps = s
            px1, py1, px2, py2 = line[0]

    # Display the frame with text
    if belt_detected_frames_count > 0:
        cv2.putText(beltframe, "Belt Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        belt_detected_frames_count -= 1
    else:
        # Increment the frame counter when no belt is detected
        no_belt_frames_count += 1
        cv2.putText(beltframe, "No Belt Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Seat Belt Detection", beltframe)

    # Check for no belt condition and initiate Twilio phone call
    if no_belt_frames_count >= no_belt_frames_threshold:
        # Make a phone call alert using Twilio
        call = twilio_client.calls.create(
            to="+17797752514",
            from_="+18888617564",
            url='http://demo.twilio.com/docs/voice.xml'  # A TwiML URL for the call (you can customize this)
        )

        # Print the call SID for reference
        #print("Phone call SID:", call.sid)

        # Break the loop after initiating the Twilio call
        break

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()