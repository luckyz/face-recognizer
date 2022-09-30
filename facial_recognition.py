import cv2
import os

data_dir = "data"
image_paths = os.listdir(data_dir)
model_dir = "model"

# model_name = "modelEigenFace.xml"
# model_name = "modelFisherFace.xml"
model_name = "modelLBPHFace.xml"
model_path = os.path.join(model_dir, model_name)
print("image_paths:", image_paths)

# face_recognizer = cv2.face.EigenFaceRecognizer_create()
# face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# reading model
face_recognizer.read(model_path)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap = cv2.VideoCapture("video.mp4")

face_classif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
	ret, frame = cap.read()
	if ret == False: break
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	aux_frame = gray.copy()

	faces = face_classif.detectMultiScale(gray, 1.3, 5)

	for (x, y, w, h) in faces:
		face = aux_frame[y:y+h, x:x+w]
		face = cv2.resize(face, (150, 150), interpolation=cv2.INTER_CUBIC)
		result = face_recognizer.predict(face)

		cv2.putText(frame, "{}".format(result), (x, y-5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

		# EigenFaces
		if result[1] < 5700:
			cv2.putText(frame, "{}".format(image_paths[result[0]]),(x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		else:
			cv2.putText(frame, "Unknown", (x, y-20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
			cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)

		# # FisherFace
		# if result[1] < 500:
		# 	cv2.putText(frame, "{}".format(image_paths[result[0]]), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
		# 	cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		# else:
		# 	cv2.putText(frame, "Unknown", (x, y-20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
		# 	cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)

		# # LBPHFace
		# if result[1] < 70:
		# 	cv2.putText(frame, "{}".format(image_paths[result[0]].capitalize()), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
		# 	cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
		# else:
		# 	cv2.putText(frame, "Unknown", (x, y-20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
		# 	cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)
		
	cv2.imshow("frame", frame)
	k = cv2.waitKey(1)
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()
