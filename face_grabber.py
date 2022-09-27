import cv2
import os
import imutils

person_name = input("> Type person name: ").capitalize()
data_path = "data"
person_path = os.path.join(data_path, person_name.lower())

if not os.path.exists(person_path):
	print("Folder created successfully: ", person_path)
	os.makedirs(person_path)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#cap = cv2.VideoCapture("video.mp4")

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
count = 0

while True:
	ret, frame = cap.read()
	if ret == False: break
	frame =  imutils.resize(frame, width=640)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	aux_frame = frame.copy()

	faces = faceClassif.detectMultiScale(gray, 1.3, 5)

	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		rostro = aux_frame[y:y+h, x:x+w]
		rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
		cv2.imwrite(os.path.join(person_path, "face_{}.jpg").format(count),rostro)
		count += 1
	cv2.imshow("frame", frame)

	k =  cv2.waitKey(1)
	if k == 27 or count >= 300:
		break

cap.release()
cv2.destroyAllWindows()