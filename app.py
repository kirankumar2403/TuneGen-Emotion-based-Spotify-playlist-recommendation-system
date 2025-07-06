from flask import Flask, render_template, Response, request, redirect, url_for, session
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
app.secret_key = 'tunegen_secret_key'

# Set paths for model and haarcascade
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model.h5')
haarcascade_path = os.path.join(base_dir, 'haarcascade_frontalface_default.xml')

# Create the model architecture (same as in emotions.py)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input

# Create the model with the same architecture as in emotions.py
classifier = Sequential()
classifier.add(Input(shape=(48, 48, 1)))
classifier.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
classifier.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Flatten())
classifier.add(Dense(1024, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(7, activation='softmax'))

# Load the pre-trained weights
classifier.load_weights(model_path)

# Load face classifier
face_classifier = cv2.CascadeClassifier(haarcascade_path)

# Dictionary which assigns each label an emotion
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Global variable to store detected emotion
detected_emotion = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

def gen_frames():
    global detected_emotion
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            
            # Extract face region
            roi_gray = gray[y:y + h, x:x + w]
            
            # Resize and prepare for prediction
            try:
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi = roi_gray.reshape(1, 48, 48, 1) / 255.0
                
                # Predict emotion
                prediction = classifier.predict(roi)
                maxindex = int(np.argmax(prediction))
                
                # Store detected emotion
                detected_emotion = emotion_dict[maxindex]
                
                # Display emotion text
                cv2.putText(frame, detected_emotion, (x+20, y-60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            except:
                continue
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/playlist', methods=['POST'])
def playlist():
    global detected_emotion
    
    if detected_emotion is None:
        return redirect(url_for('detection'))
    
    # Store emotion in session
    session['emotion'] = detected_emotion
    
    # Get Spotify playlist URL based on emotion
    playlist_url = get_playlist_url(detected_emotion)
    
    if playlist_url:
        return render_template('music.html', src=playlist_url)
    else:
        return render_template('404.html')

def get_playlist_url(emotion):
    # Spotify playlist URLs for different emotions based on specific mood recommendations
    # Telugu song playlists for different emotional states
    
    # Customized Telugu playlist recommendations based on emotional needs:
    # - Sad: Uplifting Telugu songs to improve mood
    # - Neutral: Peaceful Telugu songs for relaxation
    # - Angry: Calm Telugu songs to reduce tension
    # - Happy: Energetic or joyful Telugu songs to maintain positive mood
    # - Surprised: Exciting or energetic Telugu songs to match the emotion
    
    playlists = {
        # Calm Telugu songs for angry emotions
        "Angry": "https://open.spotify.com/embed/playlist/37i9dQZF1DX5q67ZpWyRrZ?utm_source=generator",
        
        # Telugu songs for disgusted emotion
        "Disgusted": "https://open.spotify.com/embed/playlist/37i9dQZF1DX5OepaGriAIm?utm_source=generator",
        
        # Telugu songs for fearful emotion
        "Fearful": "https://open.spotify.com/embed/playlist/37i9dQZF1DX5OepaGriAIm?utm_source=generator",
        
        # Energetic/joyful Telugu songs for happy emotions
        "Happy": "https://open.spotify.com/embed/playlist/37i9dQZF1DX1i3hvzHpcQV?utm_source=generator",
        
        # Peaceful Telugu songs for neutral emotions
        "Neutral": "https://open.spotify.com/embed/playlist/37i9dQZF1DX6XE7HRLM75P?utm_source=generator",
        
        # Uplifting Telugu songs for sad emotions
        "Sad": "https://open.spotify.com/embed/playlist/37i9dQZF1DX6XE7HRLM75P?utm_source=generator",
        
        # Exciting/energetic Telugu songs for surprised emotions
        "Surprised": "https://open.spotify.com/embed/playlist/37i9dQZF1DX1i3hvzHpcQV?utm_source=generator"
    }
    
    return playlists.get(emotion)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='127.0.0.1', port=port)