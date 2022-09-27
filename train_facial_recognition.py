import cv2
import os
import numpy as np

data_dir = "data"
people_list = os.listdir(data_dir)
model_dir = "model"
model_name = "modelLBPHFace.xml"
# model_name = "modelEigenFace.xml"
# model_name = "modelFisherFace.xml"
model_path = os.path.join(model_dir, model_name)
print("People list:", people_list)

if not os.path.exists(model_dir):
	print("Folder created successfully: ", model_dir)
	os.makedirs(model_dir)

labels = []
faces_data = []
label = 0

for name_dir in people_list:
	person_path = os.path.join(data_dir, name_dir)
	print("\nReading images...\n")

	for file_name in os.listdir(person_path):
		print("Faces: ", os.path.join(name_dir, file_name))
		labels.append(label)
		faces_data.append(cv2.imread(os.path.join(person_path, file_name), 0))
		#image = cv2.imread(os.path.join(person_path, file_name), 0)
		#cv2.imshow("image", image)
		#cv2.waitKey(10)
	label += 1

#print("labels=", labels)
#print("Labels number 0: ", np.count_nonzero(np.array(labels)==0))
#print("Labels number 1: ", np.count_nonzero(np.array(labels)==1))

# methods to train recognizer
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# training face recognizer
print("\nTraining...")
face_recognizer.train(faces_data, np.array(labels))

# saving generated model
face_recognizer.write(model_path)
print("\nModel saved")