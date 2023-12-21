import cv2
import mediapipe as mp
import os
import json

#first let me load the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
#from the current directory load images, groundtruth, results , test directory
images_dir= os.path.join(current_dir, '..', 'handsdataset')
groundtruth_dir= os.path.join(current_dir, '..', 'groundtruth')
results_dir= os.path.join(current_dir,'..','results')
test_dir=os.path.join(current_dir, '..','sample_test_images_distal_segmentation')

#my idea would be store the groundtruth boxes and prediction boxes in list and then find the IOU so
annotated_label=[]
predicted_label=[]

#to get groundtruth boxes
def check_groundtruth(file):
    bboxes = []
    gt_file = os.path.join(groundtruth_dir, os.path.splitext(file)[0] + '.json')
    with open(gt_file, 'r') as f:
        gt_dict = json.load(f)
    #loop through all finger tips and append them to training data
    for i in range(5):
        points=gt_dict['shapes'][i]['points']
        x1, y1 = points[0]
        x2, y2 = points[1]
        bboxes.append([int(x1), int(y1), int(x2), int(y2)])
        #print(bboxes)
    annotated_label.append(bboxes)

#to detect fingertips in myhands dataset
def detect_fingerprint(img,file):
    # mediapipe hands module intialization
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) 
    #basically here static mode is given false because to make algorithm fast i.e, by saying it to detect whenever it has good confidence

    image = cv2.imread(img)
    c_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #THe mediapipe class only takes RGB Images
    res = hands.process(c_image)
    pbox=[]
    if res.multi_hand_landmarks:
        for h_ldmarks in res.multi_hand_landmarks:
            #There will be 21 handland marks but we want only fingertips so give their landmark ids
            for point in [4, 8, 12, 16, 20]:
                h,w,c=image.shape # h represents height, w represents width , c represents the channnel
                x = int(h_ldmarks.landmark[point].x * w)
                y = int(h_ldmarks.landmark[point].y * h)
                cv2.rectangle(image, (x-122, y-122), (x+122, y+122), (255, 255, 0), 2)
                #store the bounding box results
                pbox.append([x-122,y-122,x+122,y+122])
    predicted_label.append(pbox)
    
    #store results in results folder            
    resultfile=os.path.join(results_dir,'myhands',file)
    cv2.imwrite(resultfile,image)
    
    # Release resources
    hands.close()

    #send the file details to function to get it manual annotations
    check_groundtruth(file)

#to detect fingertips in test dataset
def detect_fingerprint_test(img,file):
    #mediapipe hands module intialization
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.01, min_tracking_confidence=0.01) 
    #basically here static mode is given false because to make algorithm fast i.e, by saying it to detect whenever it has good confidence.
    #Here as the images are cropped till fingers, So we are telling mediapipe module to dont wait for confidence to detect palm just predict it.

    image = cv2.imread(img)
    c_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #THe mediapipe class only takes RGB Images
    res = hands.process(c_image)

    if res.multi_hand_landmarks:
        for h_ldmarks in res.multi_hand_landmarks:
            for point in [8, 12, 16, 20]:
                h,w,c=image.shape # h represents height, w represents width , c represents the channnel
                x = int(h_ldmarks.landmark[point].x * w)
                y = int(h_ldmarks.landmark[point].y * h)
                cv2.rectangle(image, (x-122, y-122), (x+122, y+122), (255, 255, 0), 2)
    #store results in results folder
    resultfile=os.path.join(results_dir,'testdataset',file) 
    cv2.imwrite(resultfile,image)
    # Release resources
    hands.close()


def calculate_iou(boxA, boxB):
    # calculate intersection points
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # calculate area of intersection rectangle
    intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # calculate area of both the boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # calculate the IoU
    union=boxAArea + boxBArea - intersection
    iou = intersection / float(union)

    return iou


def get_accuracy(annotated_labels, predicted_labels):
    threshold=0.5
    no_of_images = len(annotated_labels)
    crrct_pred = 0
    i=0
    while(i<no_of_images):
        # First compare same finger  predicted box and annotated box and store that iou in list
        iou_finger = []
        for j in range(5):
            iou = calculate_iou(annotated_labels[i][j], predicted_labels[i][j])
            iou_finger.append(iou)
        
        # Average IoU score for all 5 fingers and check if its greater than threshold. If yes then its correct prediction
        avg_iou = sum(iou_finger) / len(iou_finger)
        if avg_iou >= threshold:
            crrct_pred += 1
        i=i+1
    # calculate accuracy
    accuracy = crrct_pred / no_of_images
    
    return accuracy


print("Started Detection on my handsdataset")
for file in os.listdir(images_dir):
    if file.endswith('.jpg'):
        curr_file= os.path.join(images_dir,file)
        detect_fingerprint(curr_file,file)
print("please check results/myhands folder to see results ")

print("------------------------calculating the accuracy-------------------------")
accuracy = get_accuracy(annotated_label, predicted_label)
print("Accuracy: {:.2f}%".format(accuracy * 100))



print("Started detecting on Test Dataset")
for file in os.listdir(test_dir):
    if file.endswith('.png'):
        test_file= os.path.join(test_dir,file)
        detect_fingerprint_test(test_file,file)
print("please check results/testdataset folder to see results ")

