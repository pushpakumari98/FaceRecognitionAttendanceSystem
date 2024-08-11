import streamlit as st
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def check_haarcascadefile():
    if not os.path.isfile("haarcascade_frontalface_default.xml"):
        st.error('Haarcascade file is missing. Please contact support.')
        st.stop()

def take_images(id, name):
    check_haarcascadefile()
    columns = ['SERIAL NO.', 'ID', 'NAME']
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")

    serial = 1
    if os.path.isfile("StudentDetails/StudentDetails.csv"):
        df = pd.read_csv("StudentDetails/StudentDetails.csv")
        serial = df.shape[0] + 1
    else:
        with open("StudentDetails/StudentDetails.csv", 'a+', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(columns)

    if name.isalpha() or ' ' in name:
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0

        while True:
            ret, img = cam.read()
            if not ret:
                st.error("Failed to capture image")
                break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                sampleNum += 1
                cv2.imwrite(f"TrainingImage/{name}.{serial}.{id}.{sampleNum}.jpg", gray[y:y+h, x:x+w])

                st.image(img, channels="BGR", caption=f"Taking Image {sampleNum}")

            if sampleNum > 100 or st.button("Stop"):
                break

        cam.release()
        cv2.destroyAllWindows()

        with open("StudentDetails/StudentDetails.csv", 'a+', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([serial, id, name])

        st.success(f"Images taken for ID: {id}")
    else:
        st.error("Please enter a valid name")

def train_images():
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = get_images_and_labels("TrainingImage")

    try:
        recognizer.train(faces, np.array(ids))
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        return

    recognizer.save("TrainingImageLabel/Trainner.yml")
    st.success("Profile saved successfully")

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces, ids = [], []
    
    for imagePath in image_paths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        ids.append(id)
        
    return faces, ids

def track_images():
    check_haarcascadefile()
    assure_path_exists("Attendance/")
    assure_path_exists("StudentDetails/")
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if not os.path.isfile("TrainingImageLabel/Trainner.yml"):
        st.error("Training data is missing. Please train the model first.")
        return
    
    recognizer.read("TrainingImageLabel/Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("StudentDetails/StudentDetails.csv")

    cam = cv2.VideoCapture(0)
    col_names = ['Id', 'Name', 'Date', 'Time']
    
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (225, 0, 0), 2)
            serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
            
            if conf < 50:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%I:%M:%S %p')
                name = df.loc[df['SERIAL NO.'] == serial]['NAME'].values[0]
                id = df.loc[df['SERIAL NO.'] == serial]['ID'].values[0]
                attendance = [id, name, date, timeStamp]
                st.success(f"Recognized {name}")
            else:
                id = 'Unknown'
                name = str(id)

            st.image(img, channels="BGR", caption=f"Tracking: {name}")

        if st.button("Stop"):
            break

    cam.release()
    cv2.destroyAllWindows()

    with open(f"Attendance/Attendance_{date}.csv", 'a+', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(col_names)
        writer.writerow(attendance)
    
    st.success("Attendance recorded")

def main():
    st.title("Face Recognition Based Attendance System")

    menu = ["Home", "Take Images", "Train Images", "Track Images"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.write("Welcome to the Face Recognition Based Attendance System.")
    
    elif choice == "Take Images":
        st.subheader("Take Images")
        id = st.text_input("Enter ID")
        name = st.text_input("Enter Name")
        
        if st.button("Take Images"):
            if id and name:
                take_images(id, name)
            else:
                st.warning("Please enter both ID and Name.")
    
    elif choice == "Train Images":
        st.subheader("Train Images")
        if st.button("Train"):
            train_images()
    
    elif choice == "Track Images":
        st.subheader("Track Images")
        if st.button("Track"):
            track_images()

if __name__ == "__main__":
    main()
