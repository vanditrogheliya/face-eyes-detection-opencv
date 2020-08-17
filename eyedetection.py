import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

# img = cv2.imread('face.jpg')
# img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
video = cv2.VideoCapture(0)

while video.isOpened():
    _, frame = video.read()
    img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img1, 1.1, 4)
    for (x,y,w,h) in faces:
        roi_gray = img1[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (x1,y1,w1,h1)in eyes:
            cv2.rectangle(roi_color, (x1,y1), (x1+w1,y1+h1), (0,255,255), 5)

    cv2.imshow('face', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()