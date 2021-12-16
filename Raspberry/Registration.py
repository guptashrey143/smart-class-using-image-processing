import cv2
import pandas as pd
import numpy as np
from PIL import Image
import os
from gpiozero import LED

green_led = LED(27)
red_led = LED(17)


df = pd.read_csv('student_data.csv')

# print(df.shape[0])
students_registered = df.shape[0]


## Calling the Classifiers

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

## Function to check if a given frame contains a face or not and if it does add it to the dataset

def generator(img, faceCascade, img_id, user_id):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "mix":(128,0,129)}
    coords = detector(img,faceCascade, 1.1, 10, color['green'], 'face')
    
    if len(coords) == 4:
        red_led.off()
        green_led.on()
        roi_img = img[coords[1]: coords[1] + coords[3], coords[0]: coords[0] + coords[3]]
        dataset_generator(roi_img,user_id, img_id,'B1' )
    else:
        green_led.off()
        red_led.on()
    return img

##Function to store the image in the dataset

def dataset_generator(img, id, img_id,batch):
    cv2.imwrite("/home/pi/Face Recognition/Images/user." + str(id) + "." + str(img_id) + ".jpg", img)


##Function to find the image on basis of classifier and return its coordinates

def detector(img, classifier, scaleFactor, minNeighbours, color,txt):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbours)
    coords = []
    for (x, y, w, h)in features:
        cv2.rectangle(img, (x,y),(x+w,y+h), color, 2)
        cv2.putText(img, txt, (x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x,y,w,h]
    return coords

## Function to take the details of students from the user

def get_data():
    enroll = input("Enter Enrollment No.:")
    name = input("Enter Name:")
    batch = input("Enter Batch:")
    return enroll,name,batch

## Function to write the details of the students in a csv file

def write_data(enroll, name, batch, i):
    if i > 0:
        df = pd.read_csv('student_data.csv')
        df = df.iloc[:,1:]
        df = df.append(dict(zip(df.columns,[enroll, name, batch])), ignore_index=True)
        df.to_csv('student_data.csv')
        
    else:
        data = {'Id': [enroll],'Name': [name],'Batch': [batch]}
        df = pd.DataFrame(data)
        df.to_csv('student_data.csv')


## Creating the dataset of the images of students when they get registered

VideoCapture = cv2.VideoCapture(0)

user_id,name,batch = get_data()
write_data(user_id,name,batch,students_registered)
students_registered+=1
img_id = 0
while True:
    _,img = VideoCapture.read()
    
    scale_percent = 10 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    img2 = generator(img, faceCascade,img_id,user_id)

    img_id+=1
    cv2.imshow("camera",img2)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

VideoCapture.release()
cv2.destroyAllWindows()
red_led.off()
green_led.off()



