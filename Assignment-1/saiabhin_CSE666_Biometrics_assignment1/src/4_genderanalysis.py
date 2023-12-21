import cv2
import numpy as np
import json
from deepface import DeepFace
# from deepface.detectors import build_detector

imagepath='/Users/abhinavbadinehal/saiabhin_CSE666_Biometrics_assignment1/count_faces.jpg'
gtboundingboxpath='/Users/abhinavbadinehal/saiabhin_CSE666_Biometrics_assignment1/groundtruths/4_genderanalysis_groundtruth.json'
boundingboxresultpath='/Users/abhinavbadinehal/saiabhin_CSE666_Biometrics_assignment1/results/boundingbox_results.json'

#Read the image and bounding boxes 
image= cv2.imread(imagepath)
with open(boundingboxresultpath, 'r') as read:
    boxes = json.load(read)

genderpred={}
for key,values in boxes.items():
    x=boxes[key]['x']
    y=boxes[key]['y']
    w=boxes[key]['w']
    h=boxes[key]['h']
    face = image[y:y+h, x:x+w]
    #detectedFace=build_detector('mtcnn').detectFace(face,enforce_detection=False)
    genderDetection=DeepFace.analyze(face, actions=['gender'],enforce_detection=False)
    gender_dict = genderDetection[0]['gender']
    dominant_gender = max(gender_dict, key=gender_dict.get)
    genderpred[key]={'dominant_gender':dominant_gender,'region':values}
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, str(dominant_gender), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("Gender Detected", image)
cv2.waitKey(0)

#-------------------------------------#
# lets start evaluating the model.
# The metric I am measuring for evaluation would be Intersection over union.
# #get the ground truth first.
with open(gtboundingboxpath, 'r') as read:
    groundtruth = json.load(read)

def get_iou(box1, box2):
    x1 = max(box1['x'], box2['x'])
    y1 = max(box1['y'], box2['y'])
    x2 = min(box1['x']+box1['w'], box2['x']+box2['w'])
    y2 = min(box1['y']+box1['h'], box2['y']+box2['h'])
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    boxI_area = box1["w"] * box1["h"]
    boxII_area = box2["w"] * box2["h"]
    union = boxI_area + boxII_area - intersection
    iou = intersection / float(union)
    return iou


truePositives = 0
falsePositives = 0
falseNegatives = 0


#here for each ground truth i am checking corresponding predbox with high iou
#if there then checking if its above threshold to see either tp or fp or fn
iou_threshold=0.5
for gtlabels, gtboxes in groundtruth.items():
    best_iou=0
    best_box=None
    for predlabels, predboxes in genderpred.items():
        iou=get_iou(gtboxes['region'],predboxes['region'])
        if(iou>best_iou):
            best_iou=iou
            best_box=predboxes
    if(best_iou>iou_threshold):
        if(gtboxes['dominant_gender']==best_box['dominant_gender']):
            truePositives+=1
    else:
        falseNegatives+=1
for predlabels, predboxes in genderpred.items():
    flag=False
    for gtlabels, gtboxes in groundtruth.items():
        iou=get_iou(gtboxes['region'],predboxes['region'])
        if(iou>iou_threshold):
            flag=True
            break
    if(not flag):
        falsePositives+=1


precision = truePositives / (truePositives + falsePositives)
recall = truePositives / (truePositives + falseNegatives)
f1_score = 2 * precision * recall / (precision + recall)

print("--------------Evaluations-----------")
print("Precision is: ",precision)
print("Recall is :",recall)
print("F1-score:",f1_score)


