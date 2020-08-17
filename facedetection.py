import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# img = cv2.imread('face.jpg')
# img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
video = cv2.VideoCapture(0)

while video.isOpened():
    _, frame = video.read()
    img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img1, 1.1, 4)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 3)

    cv2.imshow('face', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()