import cv2
import numpy as np
import imutils
cap = cv2.VideoCapture('VideoEntrada5.mp4')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=400)
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    color = (0, 255, 0)
    texto_estado = "Estado: No se ha detectado movimiento"
    area_pts = np.array([[50, 250], [50, 500], [350, 500], [350, 250]])
    imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
    image_area = cv2.bitwise_and(gray, gray, mask=imAux)

    fgmask=fgbg.apply(image_area)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in cnts:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            texto_estado = "Estado: Movimiento Detectado"
            color = (0, 0, 255)
    cv2.drawContours(frame,[area_pts],-1,color,2)
    cv2.putText(frame,texto_estado,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.60,color,1)
    cv2.imshow("Frame", frame)
    cv2.imshow("Gray",image_area)
    cv2.imshow("Fondo", fgmask)
    k = cv2.waitKey(70) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()