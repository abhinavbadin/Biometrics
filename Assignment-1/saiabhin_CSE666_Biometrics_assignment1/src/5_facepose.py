import cv2
import json
from headpose.detect import PoseEstimator

imagepath='/Users/abhinavbadinehal/saiabhin_CSE666_Biometrics_assignment1/count_faces.jpg'
gtboundingboxpath='/Users/abhinavbadinehal/saiabhin_CSE666_Biometrics_assignment1/groundtruths/expanalysis_groundtruth.json'
boundingboxresultpath='/Users/abhinavbadinehal/saiabhin_CSE666_Biometrics_assignment1/results/boundingbox_results.json'

#Read the image and bounding boxes 
image= cv2.imread(imagepath)
with open(boundingboxresultpath, 'r') as read:
    boxes = json.load(read)

headpose_model= PoseEstimator()
for key,values in boxes.items():
    x=boxes[key]['x']
    y=boxes[key]['y']
    w=boxes[key]['w']
    h=boxes[key]['h']
    face = image[y:y+h, x:x+w]
    #Due to pixalation or lighting there might be no clarity in faces. To avoild value error i am facing putting in try block 
    try:
        headpose_model.detect_landmarks(face, plot=True)
        roll, pitch, yaw = headpose_model.pose_from_image(face)
        if yaw<15 and pitch<10 and roll<20:
            pose="Straight"
        else:
            pose="Side"
    except ValueError as e:
        print("Error: {}".format(e))
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, pose, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("Pose Detected", image)
cv2.waitKey(0)
