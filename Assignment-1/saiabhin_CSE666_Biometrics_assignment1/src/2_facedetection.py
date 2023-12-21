import cv2
import os
import json



imagepath='/Users/abhinavbadinehal/saiabhin_CSE666_Biometrics_assignment1/count_faces.jpg'
gtboundingboxpath='/Users/abhinavbadinehal/saiabhin_CSE666_Biometrics_assignment1/groundtruths/2_groundtruth_boundingbox.json'
boundingboxresultpath='/Users/abhinavbadinehal/saiabhin_CSE666_Biometrics_assignment1/results/boundingbox_results.json'


#just read the image now 
image= cv2.imread(imagepath)

#So I am using haaarcascade classifier to detect the faces
#for that i need to take the path of haarcascade classifier
cascade = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(cascade)

#most of cv2 uses the image in gray scale so lets convert it 
gray_scale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#now using this gray scale we detect the faces in the image
faces = face_cascade.detectMultiScale(gray_scale, 1.12, 3)
detected_faces= len(faces)

#storing the detect faces in a dictionary and printing the faces
boxes={}
length=1
for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        boxes['face{}'.format(length)]={'x':int(x),'y':int(y),'w':int(w),'h':int(h)}
        #boxes['face{}'.format(length)]=list({x,y,w,h})
        length=length+1

cv2.imshow("Faces found", image)
cv2.waitKey(0)

print("No of persons present are",detected_faces)
#now i am storing that dictionary in a json file for further evaulation.
with open(boundingboxresultpath, 'w') as pr:
    json.dump(boxes, pr)

#---------------------------------------------------------------------#
#lets start evaluating the model.
# The metric I am measuring for evaluation would be Intersection over union.
# get the ground truth first.
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
    for predlabels, predboxes in boxes.items():
        iou=get_iou(gtboxes,predboxes)
        if(iou>best_iou):
            best_iou=iou
            best_box=predboxes
    if(best_iou>iou_threshold):
        truePositives+=1
    else:
        falseNegatives+=1
for predlabels, predboxes in boxes.items():
    flag=False
    for gtlabels, gtboxes in groundtruth.items():
        iou=get_iou(gtboxes,predboxes)
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
              


