import cv2
import numpy as np
import face_recognition


imagepath='/Users/abhinavbadinehal/saiabhin_CSE666_Biometrics_assignment1/count_faces.jpg'
embeddingspath='/Users/abhinavbadinehal/saiabhin_CSE666_Biometrics_assignment1/results/embeddings.npz'

#get the face according to values and store face_encodings 
face_encodings=[]
labels=[]
image = face_recognition.load_image_file(imagepath)
# Find all the face locations
face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(image, face_locations)
face_encodings=np.array(face_encodings,dtype=object)

boxes={}
len=1
for (x, y, w, h) in face_locations:
        cv2.rectangle(image, (h, x), (y, w), (255, 0, 0), 2)
        boxes['face{}'.format(len)]={'x':int(x),'y':int(y),'w':int(w),'h':int(h)}
        len=len+1
np.savez(embeddingspath, embeddings=face_encodings)



