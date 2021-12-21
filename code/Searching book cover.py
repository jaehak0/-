#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2 , glob, numpy as np
import urllib.request 
from bs4 import BeautifulSoup 
from urllib import parse
import ssl 
import webbrowser
from PIL import ImageFont, ImageDraw, Image

# 검색 설정 변수 
ratio = 0.8
MIN_MATCH = 5
# ORB 특징 검출기 생성 
detector = cv2.ORB_create()
# Flann 매칭기 객체 생성 
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1)
search_params=dict(checks=32)
matcher = cv2.FlannBasedMatcher(index_params, search_params)

# 책 표지 검색 함수 
def search(img):
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    
    results = {}
    # 책 커버 보관 디렉토리 경로 
    cover_paths = glob.glob('../img/books/*.*')
    for cover_path in cover_paths:
        #한글 이름 사진 읽기
        cover = cv2.imread(cover_path)
        
        img_array = np.fromfile(cover_path, np.uint8)
        cover = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        cv2.imshow('Searching...', cover) #검색 중인 책 표지 표시
        cv2.waitKey(5)
        gray2 = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
        kp2, desc2 = detector.detectAndCompute(gray2, None) # 특징점 검출 
        matches = matcher.knnMatch(desc1, desc2, 2) # 특징점 매칭 
        # 좋은 매칭 선별 
        good_matches = [m[0] for m in matches                     if len(m) == 2 and m[0].distance < m[1].distance * ratio]
        if len(good_matches) > MIN_MATCH: 
            # 좋은 매칭점으로 원본과 대상 영상의 좌표 구하기 
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])
            # 원근 변환 행렬 구하기 
            mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # 원근 변환 결과에서 정상치 비율 계산 
            accuracy=float(mask.sum()) / mask.size
            results[cover_path] = accuracy
    cv2.destroyWindow('Searching...')
    
    if len(results) > 0:
        results = sorted([(v,k) for (k,v) in results.items()                     if v > 0], reverse=True)
    return results

cap = cv2.VideoCapture(0)
qImg = None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('프레임이 존재하지 않습니다.')
        break
    h, w = frame.shape[:2]
    # 화면에 책을 인식할 영역 표시 
    left = w // 3
    right = (w // 3) * 2
    top = (h // 3 * 2) - (h // 4)
    bottom = (h // 3 * 2) + (h // 4)
    cv2.rectangle(frame, (left,top), (right,bottom), (255,255,255), 3)
    
    cv2.imshow('Book Searcher', frame)
    key = cv2.waitKey(10)
    if key == ord(' '): # 스페이스-바를 눌러서 사진 찍기
        qImg = frame[top:bottom , left:right]
        cv2.imshow('query', qImg)
        break
    elif key == 27 : #Esc
        break
else:
    print('카메라가 작동하지 않습니다.')
cap.release()

if qImg is not None:
    gray = cv2.cvtColor(qImg, cv2.COLOR_BGR2GRAY)
    results = search(qImg)
    if len(results) == 0 :
        print("해당하는 책 표지가 존재하지 않습니다.")
    else:
        for( i, (accuracy, cover_path)) in enumerate(results):

            print(i, cover_path, accuracy)
            if i==0:
                cover = cv2.imread(cover_path)       
                img_array = np.fromfile(cover_path, np.uint8)
                cover = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                fontpath = "fonts/gulim.ttc"
                font = ImageFont.truetype(fontpath, 20)
                img_pil = Image.fromarray(cover)
                draw = ImageDraw.Draw(img_pil)
                draw.text((10, 10),  ("Accuracy:%.2f%%\n 해당 책을 네이버로 검색하시겠습니까? y/n"%(accuracy*100)), font=font, fill=(0,255,0,0))
                cover= np.array(img_pil)
                
        cv2.imshow('Result', cover)
        print("해당 책을 네이버로 검색하시겠습니까? y/n")
        key = cv2.waitKey()
        if(key == ord('y') or key == ord('Y')): 
            print('yes')
            context = ssl._create_unverified_context() 
            search = results[0][1].split('\\')[-1].split('.')[0]
            url = 'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query='
            newUrl = url + parse.quote(search)
            webbrowser.open(newUrl)
        

cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




