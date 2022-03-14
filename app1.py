from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time
import io
from PIL import Image
import base64,cv2
import numpy as np
from flask_cors import CORS,cross_origin
import imutils

from engineio.payload import Payload

import mediapipe as mp

import requests
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def get_tolerance(str):
    if str == "low":
        tol_angle = 10
    else:
        tol_angle = 20
    return tol_angle


class Var:
    def _init_():
        counter=0
        stage = None
        t1 = t2 = time.time()
        curr_timer = time.time()
        start_time = time.time()
        times = [0] * 4
        threshtime = 2
        feedback = None
        rep_time = None
        tol_angle = get_tolerance('low')
        error = 0
        params = {"counter": counter, "timer": 0, "error": error}



def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle

def calculate_angle_lateral(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    return angle

Payload.max_decode_packets = 2048

app = Flask(__name__)
socketio = SocketIO(app,cors_allowed_origins='*' )



@app.route('/', methods=['POST', 'GET'])

def index():
    global vars
    vars = Var()
    return render_template('index.html')


def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string  = base64_string[idx+7:]
    sbuf = io.BytesIO()
    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)



@socketio.on('catch-frame')
def catch_frame(data):
    emit('response_back', data)  



@socketio.on('image')
def image(data_image):
    image = (readb64(data_image))
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
         # Recolor image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Storing curr time
        vars.curr_timer = time.time()

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # Visualize angle
            cv2.putText(image, str(angle)[:5],
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (
                            255, 255, 255), 2, cv2.LINE_AA
                        )

            # Curl counter logic
            if angle > 160 - vars.tol_angle and (vars.stage is None or vars.stage == 'up'):
                if vars.stage == 'up':
                    vars.t2 = time.time() # curr rep time
                    vars.times[(vars.counter-1)%4] = abs(vars.t2-vars.t1) # storing to track average time per rep
                    vars.rep_time = abs(vars.t2-vars.t1) # storing it to print later

                vars.t1 = time.time() # previous rep time

            if angle > 160 - vars.tol_angle:
                vars.stage = "down"

            if angle < 35 + vars.tol_angle and vars.stage == 'down':
                vars.stage = "up"
                vars.counter += 1
                # print(counter)

        except:
            pass

        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0, 0), (640, 73), (245, 117, 16), -1)

        # Rep data
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(vars.counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, 'STAGE', (140, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, vars.stage,
                    (120, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Time Data
        cv2.putText(image, 'REP TIME', (320, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(image, str(vars.rep_time)[0:4],
                    (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Feedback
        cv2.putText(image, 'FEEDBACK', (500, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        if vars.counter % 4 == 0 and vars.counter != 0:
            if (np.mean(vars.times) - vars.threshtime) > vars.threshtime/4:
                vars.feedback = 'Do Fast'
                vars.error += 1
            elif (vars.threshtime - np.mean(vars.times)) > vars.threshtime/4:
                vars.feedback = 'Do slow'
                vars.error += 1
            else:
                vars.feedback = 'Doing good'

        elif abs(vars.curr_timer-vars.t1) > 3.5: # if curr time - prev rep > 3 we say 
            if vars.stage == 'up':
                vars.feedback = 'Lower your arms'
            else:
                vars.feedback = 'Raise your arms'

        else:
            vars.feedback = None

        cv2.putText(image, vars.feedback,
                    (450, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(
                                        color=(245, 117, 66), thickness=2, circle_radius=1),
                                    mp_drawing.DrawingSpec(
                                        color=(245, 66, 230), thickness=2, circle_radius=1)
                                    )
        
        if vars.counter >= 1:
            vars.params["counter"] = vars.counter
            tim = time.time()
            vars.params["timer"] = np.round(tim - vars.start_time, 2)
            vars.params["error"] = vars.error

        imgencode = cv2.imencode('.jpeg', image,[cv2.IMWRITE_JPEG_QUALITY,40])[1]

        # base64 encode
        stringData = base64.b64encode(imgencode).decode('utf-8')
        b64_src = 'data:image/jpeg;base64,'
        stringData = b64_src + stringData

    # emit the frame back
        emit('response_back', stringData)
    


if __name__ == '__main__':
    socketio.run(app,port=9990 ,debug=True)

    