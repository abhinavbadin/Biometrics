import cv2
import face_recognition
import os
import csv

images_Path = '/Users/abhinavbadinehal/saiabhin_CSE666_Biometrics_assignment1/data'
data_Path = '/Users/abhinavbadinehal/saiabhin_CSE666_Biometrics_assignment1/data/congress.tsv'

#Loading our input biden image and their encodings
biden_image_path = '/Users/abhinavbadinehal/saiabhin_CSE666_Biometrics_assignment1/count_faces.jpg'
biden_image = face_recognition.load_image_file(biden_image_path)
bi_face_locations = face_recognition.face_locations(biden_image)
bi_encodings = face_recognition.face_encodings(biden_image, bi_face_locations)

#So at first if i see images in img folder there are 531 images.
# If i see .tsv file there are names only for 489 people.
#lets remmove some crap and store whats required only because it saves us some compuations
#So lets store image paths and their names in a dictionary
image_data = {}
with open(data_Path, 'r') as file:
    tsv = csv.reader(file, delimiter='\t')
    for row in tsv:
        filename = row[0].strip()
        name = row[1].strip() if len(row) > 1 else "Unknown"
        image_data[filename] = name


#Now looping through each embedding and then loading image embedding from dataset is time consuming 
#So lets also store encodings in a dictionary
# So loading images by their paths and encode their faces
encodings = {}
for filename, name in image_data.items():
    full_image_path = os.path.join(images_Path, filename)
    #I am putting this statements in try and except block because many times i have encountered issue with null encodings
    try:
        image = face_recognition.load_image_file(full_image_path)
        face_locations = face_recognition.face_locations(image)
        encoding = face_recognition.face_encodings(image, face_locations)[0]
        encodings[filename] = encoding
    except:
        print(f"I can't encode image: {filename}")


length=0
prediction=[]
for new_encoding,loc in zip(bi_encodings,bi_face_locations):
    flag = False
    for filename, encoding in encodings.items():
        result = face_recognition.compare_faces([encoding], new_encoding)
        if result[0]:
            flag = True
            prediction.append((loc,image_data[filename]))
            break

    if flag==False:
        prediction.append((loc,"Unknown")) 

input_image=cv2.imread(biden_image_path)
for (top, right, bottom, left), name in prediction:
    cv2.rectangle(input_image, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.putText(input_image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display output image
cv2.imshow('Labeled Faces', input_image)
cv2.waitKey(0)

