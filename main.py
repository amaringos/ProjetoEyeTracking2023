import cv2
import dlib
import numpy as np
from math import hypot

# faz reconhecimento da webcam
cap = cv2.VideoCapture(0)

# passo de verificação
if not cap.isOpened():
    print("Erro")
    exit()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 1)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 1)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

font = cv2.FONT_HERSHEY_SIMPLEX

while True:

    # faz a leitura da captura de video da webcam
    ret, frame = cap.read()

    #passo de verificação
    if not ret:
        print("Erro na leitura")
        break

    # converte o frame da camera em escala de cinzas
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        

        if  right_eye_ratio > 5.7:
            cv2.putText(frame, "PISCANDO", (10, 100), font, 3, (255, 0, 0))
        

    # abre uma janela para a webcam no tamanho 640x360
    cv2.imshow("Camera", frame)
    frame = cv2.resize(frame, (640, 360)) 
    cv2.waitKey(1)
    if cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1:
        break

# encerra 
cv2.destroyAllWindows()
cap.release()