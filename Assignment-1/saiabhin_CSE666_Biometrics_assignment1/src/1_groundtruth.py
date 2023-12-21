import cv2
import json
from deepface import DeepFace


groundtruthpath='/Users/abhinavbadinehal/saiabhin_CSE666_Biometrics_assignment1/groundtruths/count_faces_new.json'
imagepath='/Users/abhinavbadinehal/saiabhin_CSE666_Biometrics_assignment1/count_faces.jpg'
gtboundingboxpath='/Users/abhinavbadinehal/saiabhin_CSE666_Biometrics_assignment1/groundtruths/2_groundtruth_boundingbox.json'
gtexppath='/Users/abhinavbadinehal/saiabhin_CSE666_Biometrics_assignment1/groundtruths/3_expanalysis_groundtruth.json'
gtgenderpath='/Users/abhinavbadinehal/saiabhin_CSE666_Biometrics_assignment1/groundtruths/4_genderanalysis_groundtruth.json'


#Here I am loading groundtruth first
with open(groundtruthpath, 'r') as read:
    gtdict = json.load(read)
image= cv2.imread(imagepath)
num_faces = len(gtdict['shapes'])
print(num_faces)

#Changing the groundtruth structure according to bounding box results which we wil get from face detection
dict={}
for i in range(0,num_faces):
    points=gtdict['shapes'][i]['points']
    x1,y1=points[0]
    x2,y2=points[1]
    label=gtdict['shapes'][i]['label']
    dict[label]={'x':int(x1),'y':int(y1),'w':int(x2)-int(x1),'h':int(y2)-int(y1)}

with open(gtboundingboxpath, 'w') as write:
    json.dump(dict,write)


#Getting the groundtruth for Expression Analysis
emotion={}
for i in range(0,num_faces):
    label=gtdict['shapes'][i]['label']
    points=gtdict['shapes'][i]['points']
    x1,y1=points[0]
    x2,y2=points[1]
    face = image[int(y1):int(y2), int(x1):int(x2), :]
    emotionDetection=DeepFace.analyze(face, actions=['emotion'],enforce_detection=False)
    if(len(emotionDetection)>0):
        dominant_emotion=emotionDetection[0]['dominant_emotion']
        region={'x':int(x1),'y':int(y1),'w':int(x2)-int(x1),'h':int(y2)-int(y1)} 
        emotion[label]={'dominant_emotion':dominant_emotion,'region':region}

with open(gtexppath, 'w') as write:
    json.dump(emotion,write)

#Getting the groundtruth for Gender Analysis
gender={}
for i in range(0,num_faces):
    label=gtdict['shapes'][i]['label']
    points=gtdict['shapes'][i]['points']
    x1,y1=points[0]
    x2,y2=points[1]
    face = image[int(y1):int(y2), int(x1):int(x2), :]
    genderDetection=DeepFace.analyze(face, actions=['gender'],enforce_detection=False)
    if(len(genderDetection)>0):
        top_gender=genderDetection[0]['dominant_gender']
        region= {'x':int(x1),'y':int(y1),'w':int(x2)-int(x1),'h':int(y2)-int(y1)}
        gender[label]={'dominant_gender':top_gender,'region':region}

with open(gtgenderpath, 'w') as write:
    json.dump(gender,write)


