import cv2
import numpy as np   
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import streamlit as st

import os
# 안면인식 훈련모델 및 Haarcascade XML 불러오기
face_detection = cv2.CascadeClassifier(r'haarcascade\haarcascade_frontalface_default.xml')
emotion_classifier = load_model(r'final_model.h5', compile=False)
EMOTIONS = ["happiness","sadness",'surprise','neutral']

@st.cache
def take_input():
    camera = cv2.VideoCapture(0)
    while True:
    # 카메라 이미지 캡처
        ret, frame = camera.read()
        cv2.imshow('Camera', frame)
        key = cv2.waitKey(1)
        if key == ord('C'):
            break
    # 컬러에서 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        global label
        label = EMOTIONS[preds.argmax()]
        
        # 라벨 지정
        cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

        #global emotion
        # 라벨 Output
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            #global text
            text = "{}: {:.2f}%".format(emotion, prob * 100)    
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        # 2개 창 띄우기
        ## 이미지 디스플레이 ("감정인식")
        ## 감정확률 디스플레이 ("확률")
            cv2.imshow('Emotion Recognition', frame)
            cv2.imshow('Probabilities', canvas)



            # 키값 q 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        showPic = cv2.imwrite("photo.jpg",frame)
        print(showPic) 
        # 프로그램 클리어 및 창 닫기
        camera.release()
        cv2.destroyAllWindows()

# if __name__ == '__main__':
#     take_input()
# @st.cache
# def file_selector(folder_path='tempDir'):
#     filenames = os.listdir('tempDir')
#     selected_filename = st.selectbox('tempDir', filenames)
#     a = os.path.join("tempDir",image_file.name)
