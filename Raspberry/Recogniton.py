import cv2
import pandas as pd
import datetime
import numpy as np
from PIL import Image
import os
from gpiozero import LED

green_led = LED(27)
red_led = LED(17)


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.yml")


def is_class_time(weekday,st,et):
    
    x = datetime.datetime.now()
    c_weekday = x.strftime("%A")
    
   
    time = x.strftime("%T")
    c_hour = int(time.split(":")[0])
    c_min = int(time.split(":")[1])
    
    
    st_hour = int(st.split(":")[0])
    st_min = int(st.split(":")[1])
    
    
    et_hour = int(et.split(":")[0])
    et_min = int(et.split(":")[1])
    
    
    if(weekday == c_weekday):
        
        if (c_hour > st_hour) or ((c_hour == st_hour) and (c_min >= st_min)):
            
            if (c_hour < et_hour) or ((c_hour == et_hour) and (c_min < et_min)):
                return True
            
            else:
                
                return False
            
        else:
            
            return False
        
    else:
        
        return False
    
def return_batch_list():
    
    df = pd.read_csv('time chart.csv')
    
    check = 0
    pos = 0
    
    for i in df.index:
        if(is_class_time(df['Day'][i],df['Start Time'][i],df['End Time'][i])):
            check = 1
            pos = i
            break
    
    if check ==1:    
        return True, df['Batch'][pos]
    else:
        return False,''

def does_batch_belong(current_batches, stu_batch):
    batch = "'" + stu_batch +"'"
    if current_batches.find(batch) != -1:
        return True
    else:
        return False
    
def write_present(enroll, name):
    present_list = []
    time = datetime.datetime.now()
    entrytime = time.strftime('%T')
    df = pd.read_csv('present_students.csv')
    df = df.iloc[:,1:]
    for i in df.index:
        present_list.append(df['Id'][i])
    check = 0
    for i in present_list:
        if i == enroll:
            check = 1
            break
    if check == 0:
        df = df.append(dict(zip(df.columns,[enroll, name, entrytime])), ignore_index=True)
        df.to_csv('present_students.csv')

## Function to return the name of student based on the enrollment no
def user_name(user_id):
    df = pd.read_csv('student_data.csv')
    is_class, batch_list = return_batch_list()
    if is_class:
        for i in df.index:
            if df['Id'][i] == user_id:
                name = df['Name'][i]
#                 return name
                if does_batch_belong(batch_list,df['Batch'][i]):
                    name = df['Name'][i]
                    write_present(user_id,name)
                    return name
                else:
                    return 'Wrong Batch'
    else:
        return 'no_class'
    
# def get_name(user_id):
#     df = pd.read_csv('student_data.csv')
#     for i in df.index:
#             if df['Id'][i] == user_id:
#                 name = df['Name'][i]
# #                 return name
#                 if does_batch_belong(batch_list,df['Batch'][i]):
#                     name = df['Name'][i]
#                     write_present(user_id,name)
#                     return name
#                 else:
#                     return 'Wrong Batch'

def recognizer_boundary(img, classifier, scaleFactor, minNeighbours, color, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbours)
    coords = []
    if len(coords) == 0:
        red_led.off()
        green_led.off()
    for (x, y, w, h)in features:
        cv2.rectangle(img, (x,y),(x+w,y+h), color, 2)
        user_id, _ = clf.predict(gray_img[y:y+h, x:x+w])
        name = user_name(user_id)
        if(name == 'Wrong Batch'):
            red_led.on()
        else:
            green_led.on()
#         print(name)
        #name = get_name(user_id)
        cv2.putText(img, name, (x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x,y,w,h]
    return coords



def recognize(img, clf, faceCascade):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "mix":(128,0,129)}
    coords = recognizer_boundary(img, faceCascade, 1.1, 10, color["green"], clf)
    return img

VideoCapture = cv2.VideoCapture(0)

while True:
    _,img = VideoCapture.read()
    scale_percent = 10 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img2 = recognize(img, clf, faceCascade)
    cv2.imshow("camera",img2)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

VideoCapture.release()
cv2.destroyAllWindows()