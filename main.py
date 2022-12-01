# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import audioop
import pyaudio
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import face_recognition
import pyttsx3

def mse(img1, img2):
   diff = cv.subtract(img1, img2)
   err = np.sum(diff**2)
   return err

def detect_alarm():
    while True:
        chunk = 1024
        list_frames = []
        img_list = ["caroline.PNG", "christine.jpeg", "eric.jpeg", "mom.jpeg"]
        people = ["Caroline", "Christine", "Eric", "Emily"]
        cat_img_list = ["appa.PNG", "evie.png", "zellie.jpeg"]
        encodings = []
        people_recognized = []

        catFaceCascade = cv.CascadeClassifier('haarcascade_frontalcatface.xml')

        p = pyaudio.PyAudio()

        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=44100,
                        input=True,
                        frames_per_buffer=chunk)

        data = stream.read(chunk)

        rms = audioop.rms(data, 1)
        if rms>30:
            print("Person Detected.", rms)
            cv.cvtColor = cv.COLOR_BGR2HSV
            cap = cv.VideoCapture(0)
            if not cap.isOpened():
                print("Cannot open camera")
                exit()
            for i in range(1):
                # Capture frame-by-frame
                ret, frame = cap.read()
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                list_frames.append(frame)
            plt.imshow(list_frames[0], interpolation='nearest')
            unknown_image = list_frames[0]
            plt.imshow(unknown_image)
            plt.show()
            for img in range(len(img_list)):
                known_image1 = face_recognition.load_image_file(img_list[img])
                if len(face_recognition.face_encodings(known_image1))>0:
                    encoding1 = face_recognition.face_encodings(known_image1)[0]
                    encodings.append(encoding1)
                    people_recognized.append(people[img])
            if len(face_recognition.face_encodings(unknown_image))>0:
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results = face_recognition.compare_faces(encodings, unknown_encoding)
                to_speak = f"Welcome {people_recognized[results.index(True)]}"
                engine = pyttsx3.init()
                engine.say(to_speak)
                engine.runAndWait()

            cap.release()
            cv.destroyAllWindows()
            return True
        else:
            pass




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    detect_alarm()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
