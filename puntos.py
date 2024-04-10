import mediapipe as mp
import cv2
import imutils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture('manejo1.mp4')

index_list =[78,63,105,66,107,336,296,334,293,380,122,196,3,51,281,248,419,351,37,8, 257]

with mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,min_detection_confidence=0.9) as face_mesh:
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame= cv2.flip(frame,1)
        height, width,_ = frame.shape
        frame_rgb= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        frame = imutils.resize(frame, width=400)


        print("Face landmarks:", results.multi_face_landmarks)
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                for index in index_list:
                    x= int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    cv2.circle(frame,(x,y),2,(255,0,255),2)
                    #mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACE_CONNECTIONS)
                    #mp_drawing.DrawingSpec(color=(0,255,255),thickness=1, circle_radius=1)
                    #mp_drawing.DrawingSpec(color=(255,0,255),thickness=1))
        cv2.imshow("Frame", frame)
        k=cv2.waitKey(1)& 0xFF
        if k== 27:
            break
cap.release()
cv2.destroyAllWindows()