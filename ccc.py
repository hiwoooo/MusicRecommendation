from fileinput import filename
import os
import shutil
from tempfile import tempdir
import cv2
import numpy as np   
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import streamlit as st
import shutil



# 안면인식 훈련모델 및 Haarcascade XML 불러오기
face_detection = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_default.xml')
emotion_classifier = load_model('./final_model.h5', compile=False)
EMOTIONS = ["happiness","sadness","surprise","neutral"]


def file1_selector(folder_path='tempDir', filename=''):
    
    filenamepath = os.path.join(folder_path, filename)
    print(filenamepath + "...")

    src = cv2.imread(filenamepath, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # 프레임에서 안면인식
    faces = face_detection.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(30,30))
    # 이미지 공간 생성
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    # 안면인식이 가능할 경우만 감정인식 진행
    if len(faces) > 0:
        # 가장 큰 이미지 Sorting
        face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = face
        # 이미지를 48x48로 사이즈 변환후 신경망에 구성
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        # 감정 예측
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        #함수 라벨 전역변수로 변환
        global label2
        label2 = EMOTIONS[preds.argmax()]
        print("ccc filename ", filename)
        print("ccc label2 ", label2)
        print("ccc pred ", EMOTIONS[preds.argmax()])        # 라벨 지정
        # cv2.putText(src, label2, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(src, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

        global emotion
        # 라벨 Output
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            #global text
            text = "{}: {:.2f}%".format(emotion, prob * 100)    
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        # 2개 창 띄우기
        # 이미지 디스플레이 ("감정인식")
        # 감정확률 디스플레이 ("확률")
        #    cv2.imshow('Emotion Recognition', src)
        #    cv2.imshow('Probabilities', canvas)
            
        # print(os.path.join(folder_path, "photo.jpg"))
        # showPic = cv2.imwrite(os.path.join(folder_path, "photo.jpg"),src)
        showPic = cv2.imwrite(folder_path + "/photo.jpg",src)
        # print(showPic)
        #cv2.destroyAllWindows()

        # if os.path.exists('tempdir'):
        #     for file in os.scandir('tempDir'):
        #         os.remove(file.path)
        #         print(file)
        


if __name__ == '__main__':
    file1_selector("tempDir","/photo.jpg")
