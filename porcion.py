import cv2
import numpy as np
import imutils
cap = cv2.VideoCapture('VideoEntrada5.mp4')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=300)
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Dibujamos un rectángulo en frame, para señalar el estado
    # del área en análisis (movimiento detectado o no detectado)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    color = (0, 255, 0)
    texto_estado = "Estado: No se ha detectado movimiento"
    # Especificamos los puntos extremos del área a analizar
    area_pts = np.array([[240, 320], [480, 320], [620, frame.shape[0]], [50, frame.shape[0]]])

    # Con ayuda de una imagen auxiliar, determinamos el área
    # sobre la cual actuará el detector de movimiento
    imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
    image_area = cv2.bitwise_and(gray, gray, mask=imAux)
    cv2.imshow('Frame', frame)
    cv2.imshow('Aux', imAux)
    cv2.imshow('Area', image_area)

    cv2.imshow("frame", frame)
    k = cv2.waitKey(70) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

