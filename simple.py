from lib2to3.pgen2.literals import test
import streamlit as st
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import capture2
import random
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from PIL import Image
import ccc
import shutil
# import pyautogui
from sympy import false
import base64


prediction_filename = ''
client_id = 'f6e77303443749bb809dd69d184d9703'
client_secret = '100a1938944043329e9a3dbdb4936fe9'
prediction_folder = "tempDir"

client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

header  = st.container()
inp = st.container()
pred = st.container()

faceCascade = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_default.xml')

def load_image(image_file):
	img = Image.open(image_file)
	return img

def main():
    st.subheader("사진을 드래그 해보세요!")    
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    
    if image_file is not None:
        for filename in os.listdir(prediction_folder):
            filepath = os.path.join(prediction_folder, filename)
            try:
                shutil.rmtree(filepath)
            except OSError:
                os.remove(filepath)
        file_details = {"filename":image_file.name, "filetype":image_file.type,
                        "filesize":image_file.size}
        st.write(file_details)
        st.image(load_image(image_file))
        with open(os.path.join(prediction_folder, image_file.name),"wb") as f:
            f.write((image_file).getbuffer())    
        st.success("File Saved")  
        print(image_file.name + " not None")
    else:
        # st.session_state['button2'] = ''
        # st.session_state['button2'] = ''
        for filename in os.listdir(prediction_folder):
            filepath = os.path.join(prediction_folder, filename)   
        print(filename + " None")

def get_track_ids(playlist_id):
    music_id_list = []
    playlist = sp.playlist(playlist_id)

    for item in playlist['tracks']['items']:
        music_track = item['track']
        music_id_list.append(music_track['id'])
        track_ids =music_id_list

    for i in range(5):
        random.shuffle(track_ids) 
        my_html = '<iframe src="https://open.spotify.com/embed/track/{}" width="300" height="100" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>'.format(track_ids[0])
        st.markdown(my_html, unsafe_allow_html=True)

def emotio():
    if(predictions == 'happiness'):
        new_title1 = '<p style="font-family:Malgun Gothic; color:Black; font-weight: bold; font-size: 25px;">기분이 좋으시네요.</p>'
        new_title2 = '<p style="font-family:Malgun Gothic; color:Black; font-weight: bold; font-size: 25px;">당신을 행복하게해 줄 노래입니다!</p>'
        new_title3 = '<p style="font-family:Malgun Gothic; color:Black; font-weight: bold; font-size: 25px;">기분 좋은 하루 보내세요^^</p>'
        st.markdown(new_title1, unsafe_allow_html=True)
        st.markdown(new_title2, unsafe_allow_html=True)
        st.markdown(new_title3, unsafe_allow_html=True)
        playlist_id = '37i9dQZF1DX5a7mln8z0Su'
        get_track_ids(playlist_id)
    
    elif (predictions == 'sadness'):    
        new_title4 = '<p style="font-family:Malgun Gothic; color:Black; font-weight: bold; font-size: 25px;">기분이 안 좋아보여요..</p>'
        new_title5 = '<p style="font-family:Malgun Gothic; color:Black; font-weight: bold; font-size: 25px;">당신을 위로해 줄 노래입니다.</p>'
        new_title6 = '<p style="font-family:Malgun Gothic; color:Black; font-weight: bold; font-size: 25px;">기분 좋은 하루 보내세요~</p>'
        st.markdown(new_title4, unsafe_allow_html=True)
        st.markdown(new_title5, unsafe_allow_html=True)
        st.markdown(new_title6, unsafe_allow_html=True)
        playlist_id = '37i9dQZF1DX5a7mln8z0Su'
        get_track_ids(playlist_id)

    elif (predictions == 'surprise'):    
        new_title7 = '<p style="font-family:Malgun Gothic; color:Black; font-weight: bold; font-size: 25px;">어이쿠! 많이 놀라셨나요?</p>'
        new_title8 = '<p style="font-family:Malgun Gothic; color:Black; font-weight: bold; font-size: 25px;">당신을 진정시켜줄 노래입니다.</p>'
        new_title9 = '<p style="font-family:Malgun Gothic; color:Black; font-weight: bold; font-size: 25px;">기분 좋은 하루 보내세요~</p>'
        st.markdown(new_title7, unsafe_allow_html=True)
        st.markdown(new_title8, unsafe_allow_html=True)
        st.markdown(new_title9, unsafe_allow_html=True)
        playlist_id = '37i9dQZF1DX5a7mln8z0Su'
        get_track_ids(playlist_id)    
        
    elif(predictions=='neutral'):
        new_title10 = '<p style="font-family:Malgun Gothic; color:Black; font-weight: bold; font-size: 25px;">별일없으세요?</p>'
        new_title11 = '<p style="font-family:Malgun Gothic; color:Black; font-weight: bold; font-size: 25px;">당신에게 행복을 줄 노래입니다.</p>'
        new_title12 = '<p style="font-family:Malgun Gothic; color:Black; font-weight: bold; font-size: 25px;">기분 좋은 하루 보내세요~</p>'
        st.markdown(new_title10, unsafe_allow_html=True)
        st.markdown(new_title11, unsafe_allow_html=True)
        st.markdown(new_title12, unsafe_allow_html=True)
        options = ['모닝커피','휴식','운동','감수성자극']
        choi = st.sidebar.multiselect('원하시는 무드를 선택하세요', options)
        
        if '모닝커피' in choi:
            st.subheader("당신을 위한 모닝커피마시며 들을 음악")
            playlist_id = '2q2wCs64TnCVmpU5GXwN5e'
            get_track_ids(playlist_id)

        if '휴식' in choi:
            st.subheader("당신을 위한 휴식하면 들을 음악")
            playlist_id= '37i9dQZF1DWSvk1AxYsbvo'
            get_track_ids(playlist_id)

        if '운동' in choi:
            st.subheader("당신을 위한 운동하며 들을 음악")
            playlist_id= '2N9Yjs4PRM01wHISvLqpWt'
            get_track_ids(playlist_id)

        if '감수성자극' in choi:
            st.subheader("당신을 위한 센치하고 싶을 때 들을 음악")
            playlist_id= '00YMfharL5qCaTiCXDAJAy'
            get_track_ids(playlist_id)




with header:
    file_ = open("bbd12ce863d6ad3a27811cca66b4b3f7.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()


    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )
    st.title('감정 분석 음악 추천 서비스')

with inp:
    col1, col2,col3 = st.columns([5,5,5])

    with col1:
        button1= st.button("얼굴이미지 캡처")
    with col2:
        button2= st.button("이미지업로드")
    with col3:
        button3=st.button("새로운 노래 추천")

if st.session_state.get('button1')!=True:
   st.session_state['button1'] = button1



if st.session_state['button1']==True:
    st.markdown("당신의 얼굴 사진을 캡처합니다.")
    st.markdown("캡처 진행을 하시려면 SHIFT+C를 눌러주세요!")
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    capture2.take_input()
    predictions = str(capture2.label)
    emotio()

if st.session_state.get('button2')!=True:
    st.session_state['button2'] = button2

if  st.session_state['button2']==True:
    if __name__ == '__main__':
        main() 
        
        faceCascade = cv2.CascadeClassifier(r'haarcascade\haarcascade_frontalface_default.xml')
        # faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        print(prediction_folder, prediction_filename, 'button2')
        for filename in os.listdir(prediction_folder):
            filepath = filename  
            ccc.file1_selector(prediction_folder, filepath)
            print(ccc.label2)
            print("--------")
            predictions = str(ccc.label2)
            emotio()

if st.session_state.get('button3')!=True:
   st.session_state['button3'] = button3

if  st.session_state['button3']==True:
   pyautogui.hotkey("ctrl","F5")
