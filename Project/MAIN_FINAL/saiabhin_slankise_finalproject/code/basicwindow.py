from tkinter import *
from tkinter import filedialog
from keras.utils import pad_sequences
import pandas as pd
import tensorflow as tf
from fastdtw import fastdtw
import numpy as np
import json
from sklearn.model_selection import train_test_split

main=Tk()
main.title("CSE666-Biometric Image Analysis Project")
main.geometry("900x600")
main.configure(background="skyblue")

global filename1
global filename2
global model


def get_max_sequence_length():

    # Reading all the signatures and loading them into final_data
    with open('file_70_8_all_cords.txt', 'r') as file:
        file_contents = file.read()
        final_data = json.loads(file_contents)

    test_size = 0.3

    train_data = {}
    test_data = {}

    for user in final_data:
        # Split Genuine signatures into training and testing sets
        gen_train, gen_test = train_test_split(final_data[user][1][0], test_size=test_size, random_state=42,
                                               stratify=None)

        # Split Forgery signatures into training and testing sets
        for_train, for_test = train_test_split(final_data[user][0][0], test_size=test_size, random_state=42,
                                               stratify=None)

        # Combine the training and testing sets for each signature type
        train_signs = [for_train, gen_train]
        test_signs = [for_test, gen_test]

        # Add the user's training and testing data to the appropriate dictionaries
        train_data[user] = train_signs
        test_data[user] = test_signs
    # Determine maximum sequence length
    max_seq_length_train = max(
        [len(signature) for user_signatures in train_data.values() for signature_set in user_signatures for
         signature in signature_set])
    max_seq_length_test = max(
        [len(signature) for user_signatures in test_data.values() for signature_set in user_signatures for signature
         in signature_set])
    max_seq_length = max(max_seq_length_train, max_seq_length_test)
    return max_seq_length

def dtw_distance(signature1, signature2):
    distance, _ = fastdtw(signature1, signature2)
    return distance

def predict_genuine_or_forgery(model, signature1, signature2, max_seq_length):
    
    # Pad the input signatures
    sig1_padded = pad_sequences([signature1], maxlen=max_seq_length, padding='post', dtype='float32')[0]
    sig2_padded = pad_sequences([signature2], maxlen=max_seq_length, padding='post', dtype='float32')[0]
    
    # Calculate the DTW distance
    dtw_dist = dtw_distance(signature1, signature2)
    dtw_input = np.array([dtw_dist]).reshape(-1, 1)
    
    # Make a prediction using the trained model
    prediction = model.predict([dtw_input, sig1_padded[np.newaxis, ...], sig2_padded[np.newaxis, ...]])
    
    # Return the result: 1 for genuine, 0 for forgery
    return 1 if prediction[0][0] > 0.5 else 0




def upload_Signature1():
    global filename1
    filename1 = filedialog.askopenfilename(initialdir="./")
    text.delete('1.0', END)
    text.insert(END,"Enrolled Signature:\n")
    if filename1:
        text.insert(END,filename1+" loaded\n")
    else:
        text.insert(END,"please select file to be loaded")

def upload_Signature2():
    global filename2
    filename2 = filedialog.askopenfilename(initialdir="./")
    text.insert(END,"\nTest Signature:\n")
    if filename2:
        text.insert(END,filename2+" loaded\n")
    else:
        text.insert(END,"please select file to do verification")

def verify_signatures():
    global model
    signatures1 = pd.read_csv(filename1)
    signatures2= pd.read_csv(filename2)
    input_signature_1 = []
    input_signature_2 = []
    for index, row in signatures1.iterrows():
        input_signature1 = [row['x'], row['y'], row['timestamp'], row['pressure'],
                           row['fingerarea'], row['velocityx'], row['velocityy'], row['accelx'],
                           row['accely'], row['accelz'], row['gyrox'], row['gyroy'], row['gyroz']]
        input_signature_1.append(input_signature1)
    for index, row in signatures2.iterrows():
        input_signature2 = [row['x'], row['y'], row['timestamp'], row['pressure'],
                           row['fingerarea'], row['velocityx'], row['velocityy'], row['accelx'],
                           row['accely'], row['accelz'], row['gyrox'], row['gyroy'], row['gyroz']]
        input_signature_2.append(input_signature2)


    max_seq_length = get_max_sequence_length()
    model = tf.keras.models.load_model('models/model_70_8_all_working.h5')
    output = predict_genuine_or_forgery(model,input_signature_1, input_signature_2,max_seq_length)
    if output == 1:
        print("Genuine")
    else:
        print("Forgery")
    



font = ('times', 15, 'bold')
title = Label(main, text='DeepSign - Deep Online Handwritten Signature Verification ',anchor=CENTER, justify=RIGHT)
title.config(bg='white', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=250,y=100)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Input Signature - 1", command=upload_Signature1)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

uploadButton2 = Button(main, text="Input Signature - 2", command=upload_Signature2)
uploadButton2.place(x=50,y=200)
uploadButton2.config(font=font1)  

resultButton = Button(main, text="Verify Signatures", command=verify_signatures)
resultButton.place(x=50,y=300)
resultButton.config(font=font1) 


main.mainloop()