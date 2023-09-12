from flask import Flask, Response, render_template
import pytesseract
import cv2
import time
from gtts import gTTS
from playsound import playsound
import speech_recognition as sr
from PIL import Image
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from keras_vggface import utils
import pandas as pd
from deepface import DeepFace
import re
import imutils
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd



app = Flask(__name__)
video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier()

#Face
face_cascade.load(cv2.samples.findFile("static/face.xml"))

#people
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#Gun
gun_cascade = cv2.CascadeClassifier('static/gun_cascade.xml')

#Car
cars_cascade = cv2.CascadeClassifier('static/cars.xml')
#NumberPlate
num_cascade = cv2.CascadeClassifier('static/numplate.xml')

# resolution of the webcam
screen_width = 1280       # try 640 if code fails
screen_height = 720

# dimension of images
image_width = 224
image_height = 224


# load the trained model
model = load_model('static/facialrecognition.h5')
model2 = load_model('static/brand.h5') 
model3 = load_model('static/flower.h5')
model4 = load_model('static/traffic.h5')
model5 = load_model('static/zebracross.h5')
model6 = load_model('static/animal.h5')
model7 = load_model('static/switch.h5')
model8 = load_model('static/toilet.h5')
model9 = load_model('static/god.h5')

#ASL
model20 = load_model('static/asl.h5')

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

##########################################

def gen2(video):
    while True:
        success,frame = video.read()
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        analysisframe = ''
        letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        hand_landmarks = result.multi_hand_landmarks
        if hand_landmarks:
            for handLMs in hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                y_min -= 20
                y_max += 20
                x_min -= 20
                x_max += 20
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
                analysisframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                analysisframe = analysisframe[y_min:y_max, x_min:x_max]
                analysisframe = cv2.resize(analysisframe,(28,28))
    
    
                nlist = []
                rows,cols = analysisframe.shape
                for i in range(rows):
                    for j in range(cols):
                        k = analysisframe[i,j]
                        nlist.append(k)
            
                datan = pd.DataFrame(nlist).T
                colname = []
                for val in range(784):
                    colname.append(val)
                datan.columns = colname
    
                pixeldata = datan.values
                pixeldata = pixeldata / 255
                pixeldata = pixeldata.reshape(-1,28,28,1)
                prediction = model20.predict(pixeldata)
                predarray = np.array(prediction[0])
                letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
                predarrayordered = sorted(predarray, reverse=True)
                high1 = predarrayordered[0]
                for key,value in letter_prediction_dict.items():
                    if value==high1:
                        cv2.putText(frame,key,(28,28),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                        print("Predicted Character 1: ", key)
                        print('Confidence 1: ', 100*value)
                    
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame2 = jpeg.tobytes()
         
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')
@app.route('/video_feed2')
def video_feed2():
    global video
    return Response(gen2(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')           
            
def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")

        audio = r.listen(source, phrase_time_limit=5)

        try:
            print("Recognizing")
            query = r.recognize_google(audio, language='en')
            print("You Said: ", query)
        except Exception as e:
            return "None"
        except sr.RequestError as e:
            return "None"
        except sr.UnknownValueError:
            return "None"
        return query

def text():
    while True:
        print("Second")
        query2 = takeCommand().lower()
        if 'capture' in query2:
            while True:
                success,frame= video.read()
                cv2.imwrite('static/test.png', frame)
                print("Done Capturing")
                break
        break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/nearby')
def index2():
    return render_template('index.html')


def gen(video):
    mytext = 'Welcome, Command me for your needs'
    language = 'en'
    speech = gTTS(text=mytext, lang=language, slow=False)
    speech.save("AI.mp3")
    playsound('AI.mp3')
    while True:
        
        query = input("Enter the command : ")
        #takeCommand().lower()
        
        #Text Detection
        if 'text' in query:
            m=50
            c=0
            mytext = 'I am trying to identify the text'
            language = 'en'
            speech = gTTS(text=mytext, lang=language, slow=False)
            speech.save("AI.mp3")
            playsound('AI.mp3')
            while True:
                c+=1
                if c<=m:
                    success,frame= video.read()
                    cv2.imwrite('static/text.png', frame)
                    ret, jpeg = cv2.imencode('.jpg', frame)

                    frame2 = jpeg.tobytes()
         
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')
                else:
                    img = cv2.imread('static/text.png')
                    print(pytesseract.image_to_string(img))
                    mytext = 'I was able to recognise the text ' + str(pytesseract.image_to_string(img))
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')

                    print("Re-Capture")   
                    break
        elif 'who are you' in query or 'what are you' in query or 'what is your name' in query or 'your name' in query or 'AI' in query or 'Artificial Intelligence' in query:
            mytext = 'I am an AI, designed to interpret real time communication for specially abled people. I hope i will be helpful at most.'
            language = 'en'
            speech = gTTS(text=mytext, lang=language, slow=False)
            speech.save("AI.mp3")
            playsound('AI.mp3')
        elif 'thank you' in query or 'helped me' in query:
            mytext = 'Your Welcome'
            language = 'en'
            speech = gTTS(text=mytext, lang=language, slow=False)
            speech.save("AI.mp3")
            playsound('AI.mp3')
        elif 'command' in query or 'commands' in query or 'calender' in query:
            mytext = "The commands are as, Recognise Face for face recognition. Staring or Watching for number of people facing you. People or Pedestrian for pedestrian detection. Text for text recognition. Weapon for weapon detection. Vehicle to detect the vehicle and its features. Colour for surrounding colour detection. Brand for detecting brand logo. Flower to detect the type of flower. Sign to detetc the road sign. Zebra cross for detecting the near by pedestrian crossing. Animal to detect the type of animal. Switch to detect the near by switch board. And Toilet to recognise the male and female restroom sign classification."
            language = 'en'
            speech = gTTS(text=mytext, lang=language, slow=False)
            speech.save("AI.mp3")
            playsound('AI.mp3')
            
        #ASL
        elif 'asl' in query or 'american sign language' in query:
             while True:
                 success,frame = video.read()
                 frame = cv2.flip(frame, 1)
                 h, w, c = frame.shape

                 analysisframe = ''
                 letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    
                 framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                 result = hands.process(framergb)
                 hand_landmarks = result.multi_hand_landmarks
                 if hand_landmarks:
                     for handLMs in hand_landmarks:
                         x_max = 0
                         y_max = 0
                         x_min = w
                         y_min = h
                         for lm in handLMs.landmark:
                             x, y = int(lm.x * w), int(lm.y * h)
                             if x > x_max:
                                 x_max = x
                             if x < x_min:
                                 x_min = x
                             if y > y_max:
                                 y_max = y
                             if y < y_min:
                                 y_min = y
                         y_min -= 20
                         y_max += 20
                         x_min -= 20
                         x_max += 20
                         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                         mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
                         analysisframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                         analysisframe = analysisframe[y_min:y_max, x_min:x_max]
                         analysisframe = cv2.resize(analysisframe,(28,28))
    
    
                         nlist = []
                         rows,cols = analysisframe.shape
                         for i in range(rows):
                             for j in range(cols):
                                 k = analysisframe[i,j]
                                 nlist.append(k)
            
                         datan = pd.DataFrame(nlist).T
                         colname = []
                         for val in range(784):
                             colname.append(val)
                         datan.columns = colname
    
                         pixeldata = datan.values
                         pixeldata = pixeldata / 255
                         pixeldata = pixeldata.reshape(-1,28,28,1)
                         prediction = model20.predict(pixeldata)
                         predarray = np.array(prediction[0])
                         letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
                         predarrayordered = sorted(predarray, reverse=True)
                         high1 = predarrayordered[0]
                         for key,value in letter_prediction_dict.items():
                             if value==high1:
                                 cv2.putText(frame,key,(28,28),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                                 print("Predicted Character 1: ", key)
                                 print('Confidence 1: ', 100*value)
                    
                     ret, jpeg = cv2.imencode('.jpg', frame)
                     frame2 = jpeg.tobytes()
         
                     yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')
        #Facial Recognition
        elif 'recognise face' in query or 'recognise faces' in query or 'nice face' in query or 'recognise space' in query or 'face recognition' in query or 'latest nice place' in query or 'nice place' in query or 'nice piece' in query:
            with open("static/facelabels.pickle", 'rb') as f:
                og_labels = pickle.load(f)
                labels = {key:value for key,value in og_labels.items()}
                print(labels)
            m = 20
            c = 0
            n = []
            f = []
            temp=[]
            # the labels for the trained model
            mytext = 'Face recognition initiated'
            language = 'en'
            speech = gTTS(text=mytext, lang=language, slow=False)
            speech.save("AI.mp3")
            playsound('AI.mp3')
            while True:
                c+=1
                if c<=m:
                    success,frame = video.read()
                    frame = cv2.flip(frame, 1)
                    
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face = face_cascade.detectMultiScale(rgb, 1.1, 5)
                    cv2.imwrite('static/emo.png',frame)
                    
                    count = 0
                    for x,y,w,h in face:
                        count +=1
                        
                        roi_rgb = rgb[y:y+h, x:x+w]
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

                        size = (image_width, image_height)
                        resized_image = cv2.resize(roi_rgb, size)
                        image_array = np.array(resized_image, "uint8")
                        img = image_array.reshape(1,image_width,image_height,3) 
                        img = img.astype('float32')
                        img /= 255
                        
                        predicted_prob = model.predict(img)

                        # Display the label
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        name = labels[predicted_prob[0].argmax()]
                        for i in predicted_prob[0]:
                            temp.append(i)
                            for j in temp:
                                if j>= 0.97:
                                    if name=='Priyasha':
                                        cv2.putText(frame, f'Name : {name}', (x,y-8),font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                    elif name=='Siva':
                                        cv2.putText(frame, f'Name : {name}', (x,y-8),font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                    elif name=='Subiksha':
                                        cv2.putText(frame, f'Name : {name}', (x,y-8),font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                    elif name=='Surya':
                                        cv2.putText(frame, f'Name : {name}', (x,y-8),font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                    
                                    f.append(name)
                            
                    cv2.imwrite('static/face.png',frame)

                    ret, jpeg = cv2.imencode('.jpg', frame)

                    frame2 = jpeg.tobytes()
         
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')
                else:
                    print("Done")
                    break

            SVRcount = 0
            SUBcount = 0
            Pcount = 0
            Scount = 0
            for i in f:
                for i in f:
                    if 'Siva'==i:
                        SVRcount += 1
                    elif 'Subiksha'==i:
                        SUBcount += 1
                    elif 'Priyasha' == i:
                        Pcount += 1
                    elif 'Surya' == i:
                        Scount += 1
            
            if count == 0:
                print("No one is watching you")
            elif count == 1:
                print("Detected " +str(count) + " person watching you ")
            else:
                print("Detected " +str(count) + " people watching you ")
            

            
            if SVRcount > 770:
                print('Detected Siva')
                mytext = 'Detected Shiva'
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3')

            elif SUBcount > 770:
                print('Detected Subiksha')
                mytext = 'Detected Subiksha'
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3')
                
            elif Pcount > 770:
                print('Detected Priyasha')
                mytext = 'Detected Priyasha'
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3')
                
            elif Scount > 770:
                print('Detected Surya')
                mytext = 'Detected Surya'
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3')
                
            else:
                print("Non Detected")
                mytext = 'I was not able to detect the person'
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3')
            
                   
            for nam in f:
                if nam not in n:
                    n.append(nam)
            
            mytext = 'According to my emotion detection, the person is'
            language = 'en'
            speech = gTTS(text=mytext, lang=language, slow=False)
            speech.save("AI.mp3")
            playsound('AI.mp3')
            
            if len(n)>0:
                emo = DeepFace.analyze(img_path='static/emo.png',enforce_detection=False)
                pt1 = []
                pt1.append(emo[0]['gender']['Woman'])
                pt1.append(emo[0]['gender']['Man'])
                g = max(pt1)
                g0 = str(g)
                m1 = re.match(r'.*?(\d)\.(\d).*?', g0)
                percentg = m1.group(0)
                
                print(emo[0]['dominant_gender'],percentg,"%") 
                mytext = "Gender is " + str(emo[0]['dominant_gender']) + percentg + "%"
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3')  
                
             
                pt = []

                pt.append(emo[0]['emotion']['angry'])
                pt.append(emo[0]['emotion']['disgust'])
                pt.append(emo[0]['emotion']['fear'])
                pt.append(emo[0]['emotion']['happy'])
                pt.append(emo[0]['emotion']['sad'])
                pt.append(emo[0]['emotion']['surprise'])
                pt.append(emo[0]['emotion']['neutral'])
            
                e = max(pt)
                e0 = str(e)
                m1 = re.match(r'.*?(\d)\.(\d).*?', e0)
                percent1 = m1.group(0)
                print(emo[0]['dominant_emotion'],percent1,"%")
                
                mytext = "Emotion is " + str(emo[0]['dominant_emotion']) + percent1 + "%"
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3') 
            
                percent2 = []

                del emo[0]['race']['latino hispanic']

                percent2.append(emo[0]['race']['asian'])
                percent2.append(emo[0]['race']['indian'])
                percent2.append(emo[0]['race']['black'])
                percent2.append(emo[0]['race']['white'])
                percent2.append(emo[0]['race']['middle eastern'])
            
                r = max(percent2)
                r0 = str(r)
                m = re.match(r'.*?(\d)\.(\d).*?', r0)
                percent3 = m.group(0)
                r2 = list(emo[0]['race'].keys())[list(emo[0]['race'].values()).index(r)]
                print(r2,percent3,'%')
                mytext = "Race is " + r2 + percent3 + "%"
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3') 
            
            else:
                emo = DeepFace.analyze(img_path='static/emo.png',enforce_detection=False)
                pt5 = []
                pt5.append(emo[0]['gender']['Woman'])
                pt5.append(emo[0]['gender']['Man'])
                g5 = max(pt5)
                g6 = str(g5)
                m1 = re.match(r'.*?(\d)\.(\d).*?', g6)
                percentg5 = m1.group(0)
                
                print("Detected a ",percentg5,"%",emo[0]['dominant_gender'],"who is unknown or covered")  
                mytext = "Detected a " + percentg5 + "%" + str(emo[0]['dominant_gender']) + " who is unknown or covered"
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3') 
            
                    
            
        #Detects how many people are looking towards you
        elif 'staring' in query or 'watching' in query:
            m = 35
            c = 0
            mytext = 'Let me see, if someone is watching you'
            language = 'en'
            speech = gTTS(text=mytext, lang=language, slow=False)
            speech.save("AI.mp3")
            playsound('AI.mp3')
            while True:
                c+=1
                if c<=m:
                    success,frame = video.read()
                    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
                    face = face_cascade.detectMultiScale(gray,1.1,5)
                    count = 0
    
                    for x,y,w,h in face:
                        count +=1
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                        cv2.putText(frame,'Face'+str(count),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,0,0))
                    cv2.imwrite('static/stare.png', frame)

                    ret, jpeg = cv2.imencode('.jpg', frame)

                    frame2 = jpeg.tobytes()
         
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')
                else:
                    print("Done")
                    break
            if count == 0:
                print("No one is watching you")
                mytext = 'No one is watching you'
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3')
            elif count == 1:
                print("Detected " +str(count) + " person watching you ")
                mytext = "Detected " +str(count) + " person watching you "
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3')
            else:
                print("Detected " +str(count) + " people watching you ")
                mytext = "Detected " +str(count) + " people watching you "
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3')
        
        #Detects the number of people infront        
        elif 'people' in query or 'pedestrian' in query:
            m = 35
            c = 0
            mytext = 'Pedestrian detection has been initiated'
            language = 'en'
            speech = gTTS(text=mytext, lang=language, slow=False)
            speech.save("AI.mp3")
            playsound('AI.mp3')
            while True:
                c+=1
                if c<=m:
                    success,frame = video.read()
                    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    cv2.imwrite('static/people.png', frame)
                    
                    if success:
                        frame = imutils.resize(frame,width=min(400,frame.shape[1]))
        
                        (region, _) = hog.detectMultiScale(frame,winStride=(4,4),padding=(4,4),scale=1.05)
        
                        for x,y,w,h in region:
                            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

                    ret, jpeg = cv2.imencode('.jpg', frame)

                    frame2 = jpeg.tobytes()
         
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')
                else:
                    print("Done")
                    break
            img = cv2.imread('static/people.png')

            img = imutils.resize(img, width=min(400,img.shape[0]))

            (region, _) = hog.detectMultiScale(img,winStride=(4,4),padding = (4,4),scale=1.07)
            
            white = []
            black = []
            red = []
            green = []
            blue = []
            
            count = 0
            for x,y,w,h in region:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                count+=1
                
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                roi = img[y:y+h,x:x+w]
                hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    
                white_lower = np.array([0,0,240],np.uint8)
                white_upper = np.array([0,15,255],np.uint8)
                white_mask = cv2.inRange(hsv,white_lower,white_upper)
    
                black_lower = np.array([0, 0, 0],np.uint8)
                black_upper = np.array([0,0,29],np.uint8)
                black_mask = cv2.inRange(hsv,black_lower,black_upper)
    
                red_lower = np.array([136,87,111],np.uint8)
                red_upper = np.array([180,255,255],np.uint8)
                red_mask = cv2.inRange(hsv,red_lower,red_upper)
    
                green_lower = np.array([25,52,72],np.uint8)
                green_upper = np.array([102,255,255],np.uint8)
                green_mask = cv2.inRange(hsv,green_lower,green_upper)
    
                blue_lower = np.array([94,80,2],np.uint8)
                blue_upper = np.array([102,255,255],np.uint8)
                blue_mask = cv2.inRange(hsv,blue_lower,blue_upper)
    
                kernal = np.ones((5,5),"uint8")
    
                white_mask = cv2.dilate(white_mask,kernal)
                res_white = cv2.bitwise_and(roi,roi,mask=white_mask)
    
                black_mask = cv2.dilate(black_mask,kernal)
                res_black = cv2.bitwise_and(roi,roi,mask=black_mask)
    
                red_mask = cv2.dilate(red_mask,kernal)
                res_red = cv2.bitwise_and(roi,roi,mask=red_mask)
    
                green_mask = cv2.dilate(green_mask,kernal)
                res_green = cv2.bitwise_and(roi,roi,mask=green_mask)
    
                blue_mask = cv2.dilate(blue_mask,kernal)
                res_blue = cv2.bitwise_and(roi,roi,mask=blue_mask)
    
                #White
                contours, hierarchy = cv2.findContours(white_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
                for pic,contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if(area>300):
                        x,y,w,h = cv2.boundingRect(contour)
                        cv2.rectangle(roi,(x,y),(x+w,y+h),(255,255,255),2)
            
                        cv2.putText(roi,"White",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255))
                        white.append("True")
            
                #Black
                contours, hierarchy = cv2.findContours(black_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
                for pic,contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if(area>300):
                        x,y,w,h = cv2.boundingRect(contour)
                        cv2.rectangle(roi,(x,y),(x+w,y+h),(0,0,0),2)
            
                        cv2.putText(roi,"Black",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0))
                        black.append("True")
            
                #Red
                contours, hierarchy = cv2.findContours(red_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
                for pic,contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if(area>300):
                        x,y,w,h = cv2.boundingRect(contour)
                        cv2.rectangle(roi,(x,y),(x+w,y+h),(0,0,255),2)
            
                        cv2.putText(roi,"Red",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255))
                        red.append("True")
            
                #Green
                contours, hierarchy = cv2.findContours(green_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
                for pic, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if(area>300):
                        x,y,w,h = cv2.boundingRect(contour)
                        cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)
            
                        cv2.putText(roi,"Green",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0))
                        green.append("True")
            
                #Blue
                contours, hierarchy = cv2.findContours(blue_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
                for pic, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if(area>300):
                        x,y,w,h = cv2.boundingRect(contour)
                        cv2.rectangle(roi,(x,y),(x+w,y+h),(255,0,0),2)
            
                        cv2.putText(roi,"Blue",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,0,0))
                        blue.append("True")
            if count == 0:
                print("No people detected")
                mytext = "No people detected"
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3')
            elif count == 1:
                print("Detected " +str(count) + " person ")
                mytext = "Detected " +str(count) + " person "
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3')
                
                if len(white)==1:
                    print("white dress detected")
                    mytext = "white dress detected"
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')
                    
                if len(black)==1:
                    print("black dress spotted")
                    mytext = "black dress detected"
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')
                    
                if len(red)==1:
                    print("red dress spotted")
                    mytext = "red dress detected"
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')
                    
                if len(green)==1:
                    print("green dress spotted")
                    mytext = "green dress detected"
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')
                    
                if len(blue)==1:
                    print("blue dress spotted")
                    mytext = "blue dress detected"
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')
                    
            else:
                print("Detected " +str(count) + " people ")
                mytext = "Detected " +str(count) + " people "
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3')
                if len(white)>=1:
                    print(str(len(white)) + " wearing white dress")
                    mytext = str(len(white)) + " wearing white dress"
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')
                    
                if len(black)>=1:
                    print(str(len(black)) + " wearning black dress")
                    mytext = str(len(black)) + " wearning black dress"
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')
                    
                if len(red)>=1:
                    print(str(len(red)) + " wearning red dress")
                    mytext = str(len(black)) + " wearning red dress"
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')
                    
                if len(green)>=1:
                    print(str(len(green)) + " wearing green dress")
                    mytext = str(len(black)) + " wearning green dress"
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')
                    
                if len(blue)>=1:
                    print(str(len(blue)) + " wearing blue dress")
                    mytext = str(len(black)) + " wearning blue dress"
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')
            
            
            
        
        #Safety    
        elif 'weapon' in query or 'armed' in query or 'gun' in query:
            m = 35
            c = 0
            firstFrame = None
            gun_exist = False
            gun_1 =[]
            mytext = 'Let me see, if you are safe'
            language = 'en'
            speech = gTTS(text=mytext, lang=language, slow=False)
            speech.save("AI.mp3")
            playsound('AI.mp3')
            while True:
                c+=1
                if c<=m:
                    success,frame = video.read()
                    frame = imutils.resize(frame, width = 500)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       
                    gun = gun_cascade.detectMultiScale(gray,1.3, 5,minSize = (100, 100))
    
                    if len(gun) > 0:
                        gun_exist = True
                        gun_1.append("True")
        
                    for (x,y,w,h) in gun:
                        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
                        roi_gray = gray[y:y+h,x:x+w]
                        roi_color = frame[y:y+h,x:x+w]
        
                    if firstFrame is None:
                        firstFrame = gray
                        continue
            
                    cv2.putText(frame,datetime.datetime.now().strftime("%H:%M:%S"),(10,frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,0,255),1)

                    ret, jpeg = cv2.imencode('.jpg', frame)

                    frame2 = jpeg.tobytes()
         
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')
                else:
                    print("Done")
                    break
                
            if len(gun_1)>11:
                if gun_exist :
                    print("The person is armed")
                    mytext = "The person is armed"
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')
                else:
                    print("The person is not armed")
                    mytext = "The person is not armed"
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')
            else:
                print("The person is not armed")
                mytext = "The person is not armed"
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3')
                
        elif 'car' in query or 'vehicle' in query:
            m = 35
            c = 0
            mytext = 'Vehicle detection has been passed'
            language = 'en'
            speech = gTTS(text=mytext, lang=language, slow=False)
            speech.save("AI.mp3")
            playsound('AI.mp3')
            while True:
                c+=1
                if c<=m:
                    success,frame = video.read()
    
                    cars = cars_cascade.detectMultiScale(frame,1.1,2)
                    count = 0
                    cv2.imwrite('static/car.png', frame)
                    
                    plate = num_cascade.detectMultiScale(frame,1.1,2)
                    for x,y,w,h in plate:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    
                    for x,y,w,h in cars:
                        count+=1
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                        cv2.putText(frame,'Vehicle '+str(count),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,0,0))
                    

                    ret, jpeg = cv2.imencode('.jpg', frame)

                    frame2 = jpeg.tobytes()
         
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')
                else:
                    print("Done")
                    break
                
            img = cv2.imread("static/car.png")
            
                   
            
            
            
            cars = cars_cascade.detectMultiScale(img,1.1,2)
            
            white = []
            black = []
            red = []
            green = []
            blue = []
            
            countV = 0
            for x,y,w,h in cars:
                countV+=1
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                roi = img[y:y+h,x:x+w]
                hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    
                white_lower = np.array([0,0,240],np.uint8)
                white_upper = np.array([0,15,255],np.uint8)
                white_mask = cv2.inRange(hsv,white_lower,white_upper)
    
                black_lower = np.array([0, 0, 0],np.uint8)
                black_upper = np.array([0,0,29],np.uint8)
                black_mask = cv2.inRange(hsv,black_lower,black_upper)
    
                red_lower = np.array([136,87,111],np.uint8)
                red_upper = np.array([180,255,255],np.uint8)
                red_mask = cv2.inRange(hsv,red_lower,red_upper)
    
                green_lower = np.array([25,52,72],np.uint8)
                green_upper = np.array([102,255,255],np.uint8)
                green_mask = cv2.inRange(hsv,green_lower,green_upper)
    
                blue_lower = np.array([94,80,2],np.uint8)
                blue_upper = np.array([102,255,255],np.uint8)
                blue_mask = cv2.inRange(hsv,blue_lower,blue_upper)
    
                kernal = np.ones((5,5),"uint8")
    
                white_mask = cv2.dilate(white_mask,kernal)
                res_white = cv2.bitwise_and(roi,roi,mask=white_mask)
    
                black_mask = cv2.dilate(black_mask,kernal)
                res_black = cv2.bitwise_and(roi,roi,mask=black_mask)
    
                red_mask = cv2.dilate(red_mask,kernal)
                res_red = cv2.bitwise_and(roi,roi,mask=red_mask)
    
                green_mask = cv2.dilate(green_mask,kernal)
                res_green = cv2.bitwise_and(roi,roi,mask=green_mask)
    
                blue_mask = cv2.dilate(blue_mask,kernal)
                res_blue = cv2.bitwise_and(roi,roi,mask=blue_mask)
    
                #White
                contours, hierarchy = cv2.findContours(white_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
                for pic,contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if(area>300):
                        x,y,w,h = cv2.boundingRect(contour)
                        cv2.rectangle(roi,(x,y),(x+w,y+h),(255,255,255),2)
            
                        cv2.putText(roi,"White",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255))
                        white.append("True")
            
                #Black
                contours, hierarchy = cv2.findContours(black_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
                for pic,contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if(area>300):
                        x,y,w,h = cv2.boundingRect(contour)
                        cv2.rectangle(roi,(x,y),(x+w,y+h),(0,0,0),2)
            
                        cv2.putText(roi,"Black",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0))
                        black.append("True")
            
                #Red
                contours, hierarchy = cv2.findContours(red_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
                for pic,contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if(area>300):
                        x,y,w,h = cv2.boundingRect(contour)
                        cv2.rectangle(roi,(x,y),(x+w,y+h),(0,0,255),2)
            
                        cv2.putText(roi,"Red",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255))
                        red.append("True")
            
                #Green
                contours, hierarchy = cv2.findContours(green_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
                for pic, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if(area>300):
                        x,y,w,h = cv2.boundingRect(contour)
                        cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)
            
                        cv2.putText(roi,"Green",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0))
                        green.append("True")
            
                #Blue
                contours, hierarchy = cv2.findContours(blue_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
                for pic, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if(area>300):
                        x,y,w,h = cv2.boundingRect(contour)
                        cv2.rectangle(roi,(x,y),(x+w,y+h),(255,0,0),2)
            
                        cv2.putText(roi,"Blue",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,0,0))
                        blue.append("True")
            
            if countV == 0:
                print("No Vehicle, the road is free to pass")
                mytext = "No Vehicle, the road is free to pass"
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3')
                
            elif countV == 1:
                print("Detected " +str(count) + " vehicle ")
                mytext = "Detected " +str(count) + " vehicle "
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3')
                
                if len(white)==1:
                    print("1 white vehicle spotted")
                    mytext = "1 white vehicle spotted"
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')
                if len(black)==1:
                    print("1 black vehicle spotted")
                    mytext = "1 black vehicle spotted"
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')
                if len(red)==1:
                    print("1 red vehicle spotted")
                    mytext = "1 red vehicle spotted"
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')
                if len(green)==1:
                    print("1 green vehicle spotted")
                    mytext = "1 green vehicle spotted"
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')
                if len(blue)==1:
                    print("1 blue vehicle spotted")
                    mytext = "1 blue vehicle spotted"
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')
            else:
                print("Detected " +str(count) + " vehicles ")
                mytext = "Detected " +str(count) + " vehicles "
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3')
                
                if len(white)>=1:
                    print(str(len(white)) + " white vehicles spotted")
                    mytext = str(len(white)) + " white vehicles spotted"
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')
                    
                if len(black)>=1:
                    print(str(len(black)) + " black vehicles spotted")
                    mytext = str(len(black)) + " black vehicles spotted"
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')
                    
                if len(red)>=1:
                    print(str(len(red)) + " red vehicles spotted")
                    mytext = str(len(red)) + " red vehicles spotted"
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')
                    
                if len(green)>=1:
                    print(str(len(green)) + " green vehicles spotted")
                    mytext = str(len(green)) + " green vehicles spotted"
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')
                    
                if len(blue)>=1:
                    print(str(len(blue)) + " blue vehicles spotted")
                    mytext = str(len(blue)) + " blue vehicles spotted"
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')
                    
        elif 'colour' in query or 'color' in query:
            m=20
            c=0
            mytext = 'Recognising the colours that surround you'
            language = 'en'
            speech = gTTS(text=mytext, lang=language, slow=False)
            speech.save("AI.mp3")
            playsound('AI.mp3')
            while True:
                c+=1
                if c<=m:
                    success,frame= video.read()
                    cv2.imwrite('static/color.png', frame)
                    ret, jpeg = cv2.imencode('.jpg', frame)

                    frame2 = jpeg.tobytes()
         
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')
                else:
                    print("Done")   
                    break   
            white = []
            black = []
            red = []
            green = []
            blue = []
            
            roi = cv2.imread('static/color.png')
            hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    
            white_lower = np.array([0,0,240],np.uint8)
            white_upper = np.array([0,15,255],np.uint8)
            white_mask = cv2.inRange(hsv,white_lower,white_upper)
    
            black_lower = np.array([0, 0, 0],np.uint8)
            black_upper = np.array([0,0,29],np.uint8)
            black_mask = cv2.inRange(hsv,black_lower,black_upper)
    
            red_lower = np.array([136,87,111],np.uint8)
            red_upper = np.array([180,255,255],np.uint8)
            red_mask = cv2.inRange(hsv,red_lower,red_upper)
    
            green_lower = np.array([25,52,72],np.uint8)
            green_upper = np.array([102,255,255],np.uint8)
            green_mask = cv2.inRange(hsv,green_lower,green_upper)
    
            blue_lower = np.array([94,80,2],np.uint8)
            blue_upper = np.array([102,255,255],np.uint8)
            blue_mask = cv2.inRange(hsv,blue_lower,blue_upper)
    
            kernal = np.ones((5,5),"uint8")
    
            white_mask = cv2.dilate(white_mask,kernal)
            res_white = cv2.bitwise_and(roi,roi,mask=white_mask)
    
            black_mask = cv2.dilate(black_mask,kernal)
            res_black = cv2.bitwise_and(roi,roi,mask=black_mask)
    
            red_mask = cv2.dilate(red_mask,kernal)
            res_red = cv2.bitwise_and(roi,roi,mask=red_mask)
    
            green_mask = cv2.dilate(green_mask,kernal)
            res_green = cv2.bitwise_and(roi,roi,mask=green_mask)
    
            blue_mask = cv2.dilate(blue_mask,kernal)
            res_blue = cv2.bitwise_and(roi,roi,mask=blue_mask)
    
            #White
            contours, hierarchy = cv2.findContours(white_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
            for pic,contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area>300):
                    x,y,w,h = cv2.boundingRect(contour)
                    cv2.rectangle(roi,(x,y),(x+w,y+h),(255,255,255),2)
            
                    cv2.putText(roi,"White",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255))
                    white.append("True")
            
            #Black
            contours, hierarchy = cv2.findContours(black_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
            for pic,contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area>300):
                    x,y,w,h = cv2.boundingRect(contour)
                    cv2.rectangle(roi,(x,y),(x+w,y+h),(0,0,0),2)
            
                    cv2.putText(roi,"Black",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0))
                    black.append("True")
            
            #Red
            contours, hierarchy = cv2.findContours(red_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
            for pic,contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area>300):
                    x,y,w,h = cv2.boundingRect(contour)
                    cv2.rectangle(roi,(x,y),(x+w,y+h),(0,0,255),2)
            
                    cv2.putText(roi,"Red",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255))
                    red.append("True")
            
            #Green
            contours, hierarchy = cv2.findContours(green_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area>300):
                    x,y,w,h = cv2.boundingRect(contour)
                    cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)
            
                    cv2.putText(roi,"Green",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0))
                    green.append("True")
            
            #Blue
            contours, hierarchy = cv2.findContours(blue_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area>300):
                    x,y,w,h = cv2.boundingRect(contour)
                    cv2.rectangle(roi,(x,y),(x+w,y+h),(255,0,0),2)
            
                    cv2.putText(roi,"Blue",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,0,0))
                    blue.append("True")
            
            wh = len(white)/5*100
            bl = len(black)/5*100
            ree = len(red)/5*100
            gr = len(green)/5*100
            blu = len(blue)/5*100
            print(f"White {wh} percentage, black {bl} percentage, red {ree} percentage, green {gr} percentage, blue {blu} percentage")
            mytext = "I was able to detect, White " + str(wh) + "percentage, black " + str(bl) + " percentage, red " + str(ree) + " percentage, green " + str(gr) + " percentage and blue" + str(blu) + " percentage"
            language = 'en'
            speech = gTTS(text=mytext, lang=language, slow=False)
            speech.save("AI.mp3")
            playsound('AI.mp3')
            
        elif 'brand' in query or 'brands' in query:
            m = 35
            c = 0
            mytext = 'Brand detection has been launched'
            language = 'en'
            speech = gTTS(text=mytext, lang=language, slow=False)
            speech.save("AI.mp3")
            playsound('AI.mp3')
            while True:
                c+=1
                if c<=m:
                    success,frame = video.read()

                    cv2.imwrite('static/brand.png', frame)
                    

                    ret, jpeg = cv2.imencode('.jpg', frame)

                    frame2 = jpeg.tobytes()
         
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')
                else:
                    print("Done")
                    break
                
            img = image.load_img("static/brand.png",target_size=(64,64))
            x = np.array(img,'uint8')
            x = np.expand_dims(x,axis=0)
            x = x.astype('float32')
            x /= 255
            p = model2.predict(x)
            pred = np.argmax(model2.predict(x))
            op = ['Acura', 'Alfa Romeo', 'Aston Martin', 'Audi', 'BMW', 'Bentley', 'Bugatti', 'Buick', 'Burger King', 'Cadillac', 'Chevrolet', 'Chrysler', 'Citroen', 'Daewoo', 'Dodge', 'Ferrari', 'Fiat', 'Ford', 'GMC', 'Genesis', 'Honda', 'Hudson', 'Hyundai', 'Infiniti', 'Jaguar', 'Jeep', 'KFC', 'Kia', 'Land Rover', 'Lexus', 'Lincoln', 'MG', 'Maserati', 'Mazda', 'McDonalds', 'Mercedes-Benz', 'Mini', 'Mitsubishi', 'Nissan', 'Oldsmobile', 'Other', 'Peugeot', 'Pontiac', 'Porsche', 'Ram Trucks', 'Renault', 'Saab', 'Starbucks', 'Studebaker', 'Subaru', 'Subway', 'Suzuki', 'Tesla', 'Toyota', 'Volkswagen', 'Volvo', 'bmw', 'hyundai', 'lexus', 'mazda', 'mercedes', 'opel', 'skoda', 'toyota', 'volkswagen']
            p0 = str(max(p[0]))
            p1 = re.match(r'.*?(\d)\.(\d).*?', p0)
            percentp = p1.group(0)
            print("I was able to predict the brand as "+op[pred]+", which was "+str(float(percentp)*100)+" percentage")
            mytext = "I was able to predict the brand as "+str(op[pred])+", which was "+str(float(percentp)*100)+" percentage"
            language = 'en'
            speech = gTTS(text=mytext, lang=language, slow=False)
            speech.save("AI.mp3")
            playsound('AI.mp3')
            
        elif 'flower' in query or 'flowers' in query:
            m = 35
            c = 0
            mytext = 'Detecting flower'
            language = 'en'
            speech = gTTS(text=mytext, lang=language, slow=False)
            speech.save("AI.mp3")
            playsound('AI.mp3')
            while True:
                c+=1
                if c<=m:
                    success,frame = video.read()

                    cv2.imwrite('static/flower.png', frame)
                    

                    ret, jpeg = cv2.imencode('.jpg', frame)

                    frame2 = jpeg.tobytes()
         
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')
                else:
                    print("Done")
                    break
                
            img = image.load_img("static/flower.png",target_size=(64,64))
            x = np.array(img,'uint8')
            x = np.expand_dims(x,axis=0)
            x = x.astype('float32')
            x /= 255
            p = model3.predict(x)
            pred = np.argmax(model3.predict(x))
            op = ['bluebell','buttercup','colts_foot','cowslip','crocus','daffodil','daisy','dandelion','fritillary','iris','lily_valley','pansy','rose','snowdrop','sunflower','tigerlily','tulip','windflower']
            p0 = str(max(p[0]))
            p1 = re.match(r'.*?(\d)\.(\d).*?', p0)
            percentp = p1.group(0)
            print("I was able to predict the flower as "+op[pred]+", which was "+str(float(percentp)*100)+" percentage")
            mytext = "I was able to predict the flower as "+str(op[pred])+", which was "+str(float(percentp)*100)+" percentage"
            language = 'en'
            speech = gTTS(text=mytext, lang=language, slow=False)
            speech.save("AI.mp3")
            playsound('AI.mp3')
            
            
        elif 'sign' in query or 'road sign' in query:
            m = 35
            c = 0
            mytext = 'Road sign board recognition has been initiated'
            language = 'en'
            speech = gTTS(text=mytext, lang=language, slow=False)
            speech.save("AI.mp3")
            playsound('AI.mp3')
            while True:
                c+=1
                if c<=m:
                    success,frame = video.read()

                    cv2.imwrite('static/traffic.png', frame)
                    

                    ret, jpeg = cv2.imencode('.jpg', frame)

                    frame2 = jpeg.tobytes()
         
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')
                else:
                    print("Done")
                    break
                
            img = image.load_img("static/traffic.png",target_size=(64,64))
            x = np.array(img,'uint8')
            x = np.expand_dims(x,axis=0)
            x = x.astype('float32')
            x /= 255
            p = model4.predict(x)
            pred = np.argmax(model4.predict(x))
            op = ['15 Speed','30 Speed','40 Speed','5 Speed','50 Speed','No Entry','56','57','60 Speed','70 Speed','80 Speed','Accident Prawn Area','Bicycle Only','Car Allowed','Caution','Compulsory Horn','Compulsory Keep Left','Compulsory Keep Right','Construction','Go Straight','Hill Down','Hill Up','Left Right Prohibited','Left Turn Prohibited','No Entry','No Horn','No Stopping or Standing','No Vechicle','Other','Overtaking Prohibited','Pedestrian Caution','Pedestrian Cross','Right Left','Railway Cross','Right Prohibited','Round About','Signal Ahead','Straight Left Prohibited','Straight Right','Straight Right Prohibited','Take Left','Take Right','Train Station Ahead','U Turn','U turn Prohibited','zebra cross']
            p0 = str(max(p[0]))
            p1 = re.match(r'.*?(\d)\.(\d).*?', p0)
            percentp = p1.group(0)
            if op[pred]=='Other':
                print("I was not able to detect any sign or zebra crossing")
                mytext = "I was not able to detect any sign or zebra crossing"
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3')
            else:
                print("I was able to predict "+op[pred]+", which was "+str(float(percentp)*100)+" percentage")
                mytext = "I was able to predict "+str(op[pred])+", which was "+str(float(percentp)*100)+" percentage"
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3')
                
        elif 'zebra crossing' in query or 'zebra cross' in query:
            m = 35
            c = 0
            mytext = 'Detecting for near by zebra cross'
            language = 'en'
            speech = gTTS(text=mytext, lang=language, slow=False)
            speech.save("AI.mp3")
            playsound('AI.mp3')
            while True:
                c+=1
                if c<=m:
                    success,frame = video.read()

                    cv2.imwrite('static/cross.png', frame)
                    

                    ret, jpeg = cv2.imencode('.jpg', frame)

                    frame2 = jpeg.tobytes()
         
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')
                else:
                    print("Done")
                    break
                
            img = image.load_img("static/cross.png",target_size=(64,64))
            x = np.array(img,'uint8')
            x = np.expand_dims(x,axis=0)
            x = x.astype('float32')
            x /= 255
            p = model5.predict(x)
            pred = np.argmax(model5.predict(x))
            op = ['Other','Zebra Cross']
            p0 = str(max(p[0]))
            p1 = re.match(r'.*?(\d)\.(\d).*?', p0)
            percentp = p1.group(0)
            if op[pred]=='Other':
                print("I was "+str(float(percentp)*100)+ " percentage not able to detect zebra cross near by")
                mytext = "I was "+str(float(percentp)*100)+ " percentage not able to detect zebra cross near by"
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3')
            else:
                print("I was able to predict "+str(op[pred])+", which was "+str(float(percentp)*100)+" percentage")
                mytext = "I was able to predict "+str(op[pred])+", which was "+str(float(percentp)*100)+" percentage"
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3')
        
        elif 'animal' in query or 'dog' in query or 'cat' in query:
            m = 35
            c = 0
            mytext = 'Let me see, if i can detect the animal'
            language = 'en'
            speech = gTTS(text=mytext, lang=language, slow=False)
            speech.save("AI.mp3")
            playsound('AI.mp3')
            while True:
                c+=1
                if c<=m:
                    success,frame = video.read()

                    cv2.imwrite('static/animal.png', frame)
                    

                    ret, jpeg = cv2.imencode('.jpg', frame)

                    frame2 = jpeg.tobytes()
         
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')
                else:
                    print("Done")
                    break
                
            img = image.load_img("static/animal.png",target_size=(64,64))
            x = np.array(img,'uint8')
            x = np.expand_dims(x,axis=0)
            x = x.astype('float32')
            x /= 255
            p = model6.predict(x)
            pred = np.argmax(model6.predict(x))
            op = ['Bear','Brown bear','Bull','Camel','Canary','Caterpillar','Cattle','Centipede','Cheetah','Chicken','Crab','Crocodile','Deer','Duck','Eagle','Elephant','Fish','Fox','Frog','Giraffe','Goat','Goldfish','Goose','Hamster','Harbor seal','Hedgehog','Hippopotamus','Horse','Jaguar','Jellyfish','Kangaroo','Koala','Ladybug','Leopard','Lion','Lizard','Lynx','Magpie','Monkey','Moths and butterflies','Mouse','Mule','Ostrich','Otter','Owl','Panda','Parrot','Penguin','Pig','Polar bear','Rabbit','Raccoon','Raven','Red panda','Rhinoceros','Scorpion','Sea lion','Sea turtle','Seahorse','Shark','Sheep','Shrimp','Snail','Snake','Sparrow','Spider','Squid','Squirrel','Starfish','Swan','Tick','Tiger','Tortoise','Turkey','Turtle','Whale','Woodpecker','Worm','Zebra','bear','cat','crow','dog','elephant','rat']
            p0 = str(max(p[0]))
            p1 = re.match(r'.*?(\d)\.(\d).*?', p0)
            percentp = p1.group(0)
            print("I was able to predict "+op[pred]+", which was "+str(float(percentp)*100) +" percentage")
            mytext = "I was able to predict "+str(op[pred])+", which was "+str(float(percentp)*100) +" percentage"
            language = 'en'
            speech = gTTS(text=mytext, lang=language, slow=False)
            speech.save("AI.mp3")
            playsound('AI.mp3')
            
        elif 'switch' in query or 'switch board' in query or 'switchboard' in query:
            m = 35
            c = 0
            mytext = 'Looking for near by switch board'
            language = 'en'
            speech = gTTS(text=mytext, lang=language, slow=False)
            speech.save("AI.mp3")
            playsound('AI.mp3')
            while True:
                c+=1
                if c<=m:
                    success,frame = video.read()

                    cv2.imwrite('static/switch.png', frame)
                    

                    ret, jpeg = cv2.imencode('.jpg', frame)

                    frame2 = jpeg.tobytes()
         
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')
                else:
                    print("Done")
                    break
                
            img = image.load_img("static/switch.png",target_size=(64,64))
            x = np.array(img,'uint8')
            x = np.expand_dims(x,axis=0)
            x = x.astype('float32')
            x /= 255
            p = model7.predict(x)
            pred = np.argmax(model7.predict(x))
            op = ['Other','Switch Board']
            p0 = str(max(p[0]))
            p1 = re.match(r'.*?(\d)\.(\d).*?', p0)
            percentp = p1.group(0)
            if op[pred]=='Other':
                print("I was "+str(float(percentp)*100)+ " percentage not able to detect switch board near by")
                mytext = "I was "+str(float(percentp)*100)+ " percentage not able to detect switch board near by"
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3')
            else:
                print("I was able to predict "+op[pred]+", which was "+str(float(percentp)*100)+" percentage")
                mytext = "I was able to predict "+str(op[pred])+", which was "+str(float(percentp)*100)+" percentage"
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3')
            
        elif 'toilet' in query or 'restroom' in query or 'rest room' in query:
            m = 35
            c = 0
            mytext = 'Let me see, if it is a male or female restroom'
            language = 'en'
            speech = gTTS(text=mytext, lang=language, slow=False)
            speech.save("AI.mp3")
            playsound('AI.mp3')
            while True:
                c+=1
                if c<=m:
                    success,frame = video.read()

                    cv2.imwrite('static/cross.png', frame)
                    

                    ret, jpeg = cv2.imencode('.jpg', frame)

                    frame2 = jpeg.tobytes()
         
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')
                else:
                    print("Done")
                    break
                
            img = image.load_img("static/cross.png",target_size=(64,64))
            x = np.array(img,'uint8')
            x = np.expand_dims(x,axis=0)
            x = x.astype('float32')
            x /= 255
            p = model8.predict(x)
            pred = np.argmax(model8.predict(x))
            op = ['Other','Female','Male']
            p0 = str(max(p[0]))
            p1 = re.match(r'.*?(\d)\.(\d).*?', p0)
            percentp = p1.group(0)
            if op[pred]=='Other':
                print("I was "+str(float(percentp)*100)+ " percentage not able to detect toilet")
                mytext = "I was "+str(float(percentp)*100)+ " percentage not able to detect toilet"
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3')
            else:
                if float(percentp)*100>50:
                    print("I was able to predict "+op[pred]+", which was "+str(float(percentp)*100)+" percentage")
                    mytext = "I was able to predict "+str(op[pred])+", which was "+str(float(percentp)*100)+" percentage"
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')
                    
                else:
                    print("I was "+str(float(percentp)*100)+ " percentage not able to detect toilet")
                    mytext = "I was "+str(float(percentp)*100)+ " percentage not able to detect toilet"
                    language = 'en'
                    speech = gTTS(text=mytext, lang=language, slow=False)
                    speech.save("AI.mp3")
                    playsound('AI.mp3')
                    
        elif 'god' in query:
            with open("static/godlabels.pickle", 'rb') as f:
                og_labels = pickle.load(f)
                labels = {key:value for key,value in og_labels.items()}
                print(labels)
            m = 20
            c = 0
            n = []
            f = []
            temp=[]
            mytext = 'Looking to identify the god'
            language = 'en'
            speech = gTTS(text=mytext, lang=language, slow=False)
            speech.save("AI.mp3")
            playsound('AI.mp3')
            while True:
                c+=1
                if c<=m:
                    success,frame = video.read()
                    frame = cv2.flip(frame, 1)
                    
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face = face_cascade.detectMultiScale(rgb, 1.1, 5)
                    
                    for x,y,w,h in face:

                        
                        roi_rgb = rgb[y:y+h, x:x+w]
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

                        size = (image_width, image_height)
                        resized_image = cv2.resize(roi_rgb, size)
                        image_array = np.array(resized_image, "uint8")
                        img = image_array.reshape(1,image_width,image_height,3) 
                        img = img.astype('float32')
                        img /= 255
                        
                        predicted_prob = model9.predict(img)

                        # Display the label
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        name = labels[predicted_prob[0].argmax()]
                        for i in predicted_prob[0]:
                            temp.append(i)
                            
                            cv2.putText(frame, f'Name : {name}', (x,y-8),font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            f.append(name)
                            
                    cv2.imwrite('static/god.png',frame)

                    ret, jpeg = cv2.imencode('.jpg', frame)

                    frame2 = jpeg.tobytes()
         
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')
                else:
                    print("Done")
                    break
            
            temp0 =[]
            img = cv2.imread("static/god.png")
            frame = cv2.flip(img, 1)
                    
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = face_cascade.detectMultiScale(rgb, 1.1, 5)

            for x,y,w,h in face:
                roi_rgb = rgb[y:y+h, x:x+w]
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

                size = (image_width, image_height)
                resized_image = cv2.resize(roi_rgb, size)
                image_array = np.array(resized_image, "uint8")
                img = image_array.reshape(1,image_width,image_height,3) 
                img = img.astype('float32')
                img /= 255
                        
                predicted_prob = model9.predict(img)

                # Display the label
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[predicted_prob[0].argmax()]
                for i in predicted_prob[0]:  
                    cv2.putText(frame, f'Name : {name}', (x,y-8),font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    temp0.append(name)
            def god(L):
                counter = 0
                num = L[0]
     
                for i in L:
                    curr_frequency = L.count(i)
                    if(curr_frequency> counter):
                        counter = curr_frequency
                        num = i
                return num
     
            if len(temp0)==0:
                print("No God near by")
            else:
                print(god(temp0))
                mytext = "I was able to detect god " + str(god(temp0))
                language = 'en'
                speech = gTTS(text=mytext, lang=language, slow=False)
                speech.save("AI.mp3")
                playsound('AI.mp3')
                

       
@app.route('/asl')
def asl():
    return render_template('asl.html')



@app.route('/video_feed')
def video_feed():
    global video
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    #app.run(threaded=True, host='192.168.154.230')
    app.run(host='0.0.0.0', port=2204, threaded=True)