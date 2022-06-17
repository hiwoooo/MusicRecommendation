# Music Recommendation

## 프로젝트 개요
본 프로젝트는 얼굴의 감정 4가지 행복, 슬픔, 놀람, 중립을 분석하여 기분에 맞게 음악을 추천해주는 시스템이다.
GPU의 도입과 빅데이터의 발전으로 인공지능이 급속도로 발전하였다.
이에 딥러닝을 Open CV 을 이용하여 감정분석, 딥러닝 모델링 및 웹서버 구축을 직접 경험을 함으로써 이해하고자 한다.
API의 이용으로 다양하고 재미있는 서비스를 만들어 내며 소비자 중심의 UI를 만들어 내는 것에 목적을 두었다.

## 프로젝트 수행 도구
- OpenCV
- Pandas, Numpy
- Tensorflow
- Keras
- Sklearn
- Streamlit
- AWS

## 데이터 수집
- Fer2013 
- AIHub(연기자지망생 감정이미지)
  - fer2013과 같은 형태로 grayscale, crop, resize하여 이미지 전처리
- Total 20297개의 이미지를 사용  
  - Happiness 7295
  - Sadness 4839
  - Surprise 3197
  - Neutral 4966
 


## 시스템 구성 흐름 및 구성
![image](https://user-images.githubusercontent.com/95407936/168747649-0a66c57e-afa3-42e9-9ed4-e3fec30035dd.png)
![image](https://user-images.githubusercontent.com/95407936/174202509-112ed594-dba6-4e80-9db7-31f1ce88120f.png)


## 음악 추천 방식
![image](https://user-images.githubusercontent.com/95407936/174202426-faf662f1-aa49-4158-8ea3-3c35746ca370.png)


## 모델 구현
- Convolution Layer + Batch Normaliztion(정규화)

![image](https://user-images.githubusercontent.com/95407936/168749946-8dd81c90-5a37-4cdf-a613-70f65f3869d6.png)

