import cv2
import numpy as np
import imutils
cap = cv2.VideoCapture('VideoEntrada5.mp4')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
bg = None
contador_frame=0
color_start = (204, 204, 0)
color_end = (204, 0, 204)
color_far = (255, 0, 0)
color_start_far = (204, 204, 0)
color_far_end = (204, 0, 204)
color_start_end = (0, 255, 255)
color_contorno = (0, 255, 0)
color_ymin = (0, 130, 255)
color_angulo = (0,255,255)
color_d = (0,255,255)
color_figura = (0, 255, 255)

while True:
    ret, frame = cap.read()
    if ret == False: break
    frame = imutils.resize(frame, width=400)
    frame = cv2.flip(frame, 1)
    frameAux = frame.copy()
    if ret:
        contador_frame += 1
    else:
        break
    if bg is not None:
        ROI = frame[50:480, 50:350]
        cv2.rectangle(frame, (50 - 2, 50 - 2), (350 + 2, 480 + 2), color_figura, 1)
        grayROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        bgROI = bg[50:480, 50:350]
        dif = cv2.absdiff(grayROI, bgROI)
        _, th = cv2.threshold(dif, 30, 255, cv2.THRESH_BINARY)
        th = cv2.medianBlur(th, 7)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

        for cnt in cnts:
            M = cv2.moments(cnt)
            if M["m00"] == 0: M["m00"] = 1
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            cv2.circle(ROI, tuple([x, y]), 5, (0, 255, 0), -1)
            ymin = cnt.min(axis=1)
            cv2.circle(ROI, tuple(ymin[0]), 5, color_ymin, -1)
            hull1 = cv2.convexHull(cnt)
            cv2.drawContours(ROI, [hull1], 0, color_contorno, 2)
            hull2 = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull2)

            if defects is not None:

                inicio = []
                fin = []
                contador = 0

                for i in range(defects.shape[0]):

                    s, e, f, d = defects[i, 0]
                    start = cnt[s][0]
                    end = cnt[e][0]
                    far = cnt[f][0]
                    a = np.linalg.norm(far - end)
                    b = np.linalg.norm(far - start)
                    c = np.linalg.norm(start - end)

                    angulo = np.arccos((np.power(a, 2) + np.power(b, 2) - np.power(c, 2)) / (2 * a * b))
                    angulo = np.degrees(angulo)
                    angulo = int(angulo)

                    if np.linalg.norm(start - end) > 20 and angulo < 90 and d > 12000:
                        inicio.append(start)
                        fin.append(end)
                        cv2.circle(ROI, tuple(start), 5, color_start, 2)
                        cv2.circle(ROI, tuple(end), 5, color_end, 2)
                        cv2.circle(ROI, tuple(far), 7, color_far, -1)

                    cv2.line(ROI,tuple(start),tuple(far),color_start_far,2)
                    cv2.line(ROI,tuple(far),tuple(end),color_far_end,2)
                    cv2.line(ROI,tuple(start),tuple(end),color_start_end,2)

                if len(inicio) == 0:
                    minY = np.linalg.norm(ymin[0] - [x, y])
                    if minY >= 110:
                        contador = contador + 1
                        cv2.putText(ROI, '{}'.format(contador), tuple(ymin[0]), 1, 1.7, (color_figura), 1, cv2.LINE_AA)

                for i in range(len(inicio)):
                    contador = contador + 1
                    cv2.putText(ROI, '{}'.format(contador), tuple(inicio[i]), 1, 1.7, (color_figura), 1, cv2.LINE_AA)
                    if i == len(inicio) - 1:
                        contador = contador + 1
                        cv2.putText(ROI, '{}'.format(contador), tuple(fin[i]), 1, 1.7, (color_figura), 1, cv2.LINE_AA)

                if contador==0:
                    cv2.putText(frame, 'Posicion correcta', (100, 45), 1, 1, (color_figura), 2, cv2.LINE_4)
                else:
                    cv2.putText(frame, 'Posicion incorrecta', (100, 45), 1, 1, (color_figura), 2, cv2.LINE_4)
        cv2.imshow('Deteccion', th)
    cv2.imshow('Video', frame)
    k = cv2.waitKey(70) & 0xFF
    if contador_frame == 20:
        bg = cv2.cvtColor(frameAux, cv2.COLOR_BGR2GRAY)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()