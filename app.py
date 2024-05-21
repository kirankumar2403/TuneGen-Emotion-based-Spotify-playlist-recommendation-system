from flask import Flask, render_template, request, Response, redirect, url_for
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import time


app=Flask(__name__)
camera=cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier('D:/COMSATS/5th Semester/Human Computer Interaction/Project\Music Recommender System/haarcascade_frontalface_default.xml')
classifier =load_model('D:/COMSATS/5th Semester/Human Computer Interaction/Project\Music Recommender System/model.h5')
emotion_labels = ['Angry','Fear','Happy','Neutral', 'Sad', 'Surprise']

def generate_frames():
    camera=cv2.VideoCapture(0) 
    while True:
        success,frame=camera.read()
        if not success:
            break
        else:
            capture_duration = 5
            start_time = time.time()
            while( int(time.time() - start_time) < capture_duration ):
                _, frame = camera.read()
                frame=cv2.flip(frame,1)
                labels = []
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                faces = face_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

                for (x,y,w,h) in faces:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
                    roi_gray = gray[y:y+h,x:x+w]
                    roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

                    if np.sum([roi_gray])!=0:
                        roi = roi_gray.astype('float')/255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi,axis=0)
                        prediction = classifier.predict(roi)[0]
                        global label
                        label=emotion_labels[prediction.argmax()]
                        label_position = (x,y)
                        cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                        print(label)
                ret,buffer=cv2.imencode('.jpg',frame)
                frame=buffer.tobytes()
                cv2.waitKey(0)
                yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n'+ frame + b'\r\n\r\n')
@app.route('/')
def main():
    return render_template('index.html')

@app.route('/detection')
def index():
    return render_template('detection.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


######### Code for displaying Videos or Music ##################
@app.route('/playlist',methods = ["POST", "GET"])

def playlistmanager():

        try:
            label
        except NameError:
            return render_template('404.html')
        else:
            
            input=[]
            input=label
            print(input)
                    ###playlist selection###
            if input=='Happy':
                   return render_template('music.html',input="Happy", src="https://open.spotify.com/embed/playlist/0xKR9q0f6Lp9gUzpJtPJIr?utm_source=generator&theme=0")
               
            elif input=='Sad':
                   return render_template('music.html',input="Sad", src="https://open.spotify.com/embed/playlist/6MMSkLCHEZQSMEdAKG6ewq?utm_source=generator" )
                                                                                        
            elif input=='Neutral':
                   return render_template('music.html',input="Neutral", src="https://open.spotify.com/embed/playlist/2GFblcGwMfYLY9QNsC6jTp?utm_source=generator")
               
            elif input=='Angry':
                   return render_template('music.html',input="Angry", src="https://open.spotify.com/embed/playlist/11Wrk6FFsSHorOFsCcj4J8?utm_source=generator")
               
            elif input=='Surprise':
                   return render_template('music.html',input="Surprice", src="https://open.spotify.com/embed/playlist/5xWyRko1R0vU8H2YvdMdeu?utm_source=generator")
               
            elif input=='Fear':
                   return render_template('music.html',input="Fear", src="https://open.spotify.com/embed/playlist/2tswsTtlpSXvg4dvxgdu6l?utm_source=generator")
            else:
                return Response("Enter valid keyword")

            
if __name__ == '__main__':
    app.run(debug=True)