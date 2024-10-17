import cv2
import os
import time
import tensorflow as tf
from PIL import Image, ImageEnhance
import numpy as np
# from nltk.tokenize import PunktSentenceTokenizer
# from nltk.tokenize import WordPunctTokenizer
import imutils
import streamlit as st
# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
import warnings
from imutils.contours import sort_contours
# warnings.filterwarnings('ignore')
model_test=tf.keras.models.load_model('newocr.model')
# st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Handwritten Classification Using OCR")
st.subheader("Upload Image Below")
imag=st.file_uploader("Image",type=['png'])
if st.button("Predict Now"):
    # try:
    # # st.image(img)
    file_bytes = np.asarray(bytearray(imag.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(3,3),0)
    abc=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,9)
    invertion=255-abc
    # Dilation=cv2.dilate(invertion,np.ones((3,3))) # Depending upon the 3,3 or 4,4 the text becomes more bold then after a point it cannot be recognized properly
    def preprocess_img(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 7)
        # edges = cv2.Canny(blur, 40, 150)
        # dilation = cv2.dilate(edges, np.ones((3,3)))  
        return gray,blur
    def extract_roi(img):
        roi = img[y:y + h, x:x + w]
        return roi
    def thresholding(img):
        thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        return thresh
    def resize_img(img, w, h):
        if w > h:
            resized = imutils.resize(img, width = 28)
        else:
            resized = imutils.resize(img, height = 28)
        (h, w) = resized.shape
        dX = int(max(0, 28 - w) / 2.0)
        dY = int(max(0, 28 - h) / 2.0)
        # """
        # Padding, in the context of image processing, refers to the addition of extra pixels around the edges of an image. 
        # It is often required to achieve specific image dimensions, aspect ratios, or to preserve important information 
        # near the edges of the image. In the code you provided, padding is used to ensure that the resized image is exactly 
        # 28x28 pixels and centered within that canvas.
        # When dX is negative (i.e., w is greater than or equal to 28), horizontal padding is not required because the 
        # image is already wide enough or wider than the desired 28 pixels. In this case, dX is negative, indicating that 
        # there is no need to add padding on the left and right sides of the image.
        # When dX is positive (i.e., w is less than 28), horizontal padding is required to make the image wider and
        # center it within a 28x28 canvas. In this case, dX is positive, indicating that padding is needed on both sides of the 
        # image to achieve the desired width.
        # The same logic applies to vertical padding (dY):
        # When dY is negative (i.e., h is greater than or equal to 28), vertical padding is not required because the image is 
        # already tall enough or taller than the desired 28 pixels.
        # When dY is positive (i.e., h is less than 28), vertical padding is required to make the image taller and center 
        # it within a 28x28 canvas.
        # """
        filled = cv2.copyMakeBorder(resized, top=dY, bottom=dY, right=dX, left=dX, borderType=cv2.BORDER_CONSTANT, value = (0,0,0))
        filled = cv2.resize(filled, (28,28))
        return filled
    def normalization(img):
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis = -1)
        return img
    def find_contours(img):
        conts=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # There are two boundries in letter O above internal and external but we only need to worry about the external
        conts=imutils.grab_contours(conts)
        conts=sort_contours(conts,method='left-to-right')[0]# to specify it is the first parameter
        return conts
    k=find_contours(invertion)
    # for c in k:
    #     (x,y,w,h)=cv2.boundingRect(c)
    imgcopy=img.copy()
    minw,maxw=4,160 # Static coordinate to be verified
    minh,maxh=14,140
    digits='0123456789'
    alpha='ABCDEFGHIJHLMNOPARSTUVWXYZQBDEFGKNQRT'
    listc=digits+alpha
    listc=[i for i in listc]
    text=""
    characters=[]
    def process_box(gray, x, y, w, h):
        roi = extract_roi(gray)
        thresh = thresholding(roi)
        (h, w) = thresh.shape
        resized = resize_img(thresh, w, h)
        normalized = normalization(resized)
        characters.append((normalized, (x, y, w, h))) 
        # """
        # So here in characters[0] it stores information of the image ie pixels and characters[1] stores the x,y w,h ie position of image
        # """
    min_h, max_h = 2, 200
    min_w, max_w = 6, 200
    digits_2 = 'OI234S678g'
    letters_2 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    l2 = digits_2 + letters_2
    l2 = [l for l in l2]
    # height, width = img.shape[:2]
    # contours_size = sorted(k, key=cv2.contourArea, reverse=True)
    # def draw_img(img_cp, character):
    #     cv2.rectangle(img_cp, (x, y), (x + w, y + h), (255, 100, 0), 2)
    #     cv2.putText(img_cp, character, (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
    # def prediction(predictions, characters_list):
    #     i = np.argmax(predictions)
    #     probability = predictions[i]
    #     character = characters_list[i]
    #     return i, probability, character
    # for c in contours_size:
    #     (x1, y1, w1, h1) = cv2.boundingRect(c)

    #     if (w1 >= (width / 2)) and (h1 >= height / 2):
    #         cut_off = 8
    #         cut_img = img[y+cut_off:y1 + h1 - cut_off, x1+cut_off:x1+ w1 - cut_off]
    #         grayn, processed_img = preprocess_img(img)
    #         knew=find_contours(processed_img.copy())
    #         for c in k:
    #             #print(c)
    #                 (x, y, w, h) = cv2.boundingRect(c)
    #                 if (w >= min_w and w <= max_w) and (h >= min_h and h <= max_h):
    #                     process_box(gray, x, y, w, h)
    #             # print(characters[0])
    #         boxes = [box[1] for box in characters] # Stores all boundries ie boxes
    #         pixels = np.array([pixel[0] for pixel in characters], dtype = 'float32') # Stores all Pixels required
    #         # print(pixels.shape)
    #         predictions=model_test.predict(pixels)
    #         img_cp=img.copy
    #         for (pred, (x, y, w, h)) in zip(predictions, boxes):
    #             i, probability, character = prediction(pred, listc)
    #             draw_img(img_cp, character)




    for c in k:
    #print(c)
        (x, y, w, h) = cv2.boundingRect(c)
        if (w >= min_w and w <= max_w) and (h >= min_h and h <= max_h):
            process_box(gray, x, y, w, h)
    # print(characters[0])
    boxes = [box[1] for box in characters] # Stores all boundries ie boxes
    pixels = np.array([pixel[0] for pixel in characters], dtype = 'float32') # Stores all Pixels required
    # print(pixels.shape)
    img_copy = img.copy()
    l1=["0"]
    r=0
    text1=""
    charlist=[]
    predictions=model_test.predict(pixels)
    k=0
    for (prediction, (x, y, w, h)) in zip(predictions, boxes):
        i = np.argmax(prediction)
        #print(i)
        probability = prediction[i]
        #print(probability)
        character = listc[i]
        charlist.append(character)
        text+=listc[i]
        # st.text(boxes[k])
        text1+="T"
        # t=[]
        # o=0
        # for i in range (len(boxes)):
        #     if 60>boxes[i][2]>0:
        #         t.append(character)
        # o+=1
        #print(character)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255,100,0), 2)
        cv2.putText(img_copy, character, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        st.image(img_copy)
        print(character, ' -> ', probability * 100)
        r+=1
        k+=1
        
    # for c in k:
        # (x,y,w,h)=cv2.boundingRect(c)
        # # print(x,y,w,h)
        # if(w>minw and w<maxw and h>minh and h<maxh):
        #     process_box(gray,x,y,w,h)
        #     imag=cv2.resize(thr,(28,28))
        #     imag=imag.astype('float32')/255
        #     imag=np.expand_dims(imag,axis=-1)
        #     imag=np.reshape(imag,(1,28,28,1))
        #     m=model_test.predict(imag)
        #     r=np.argmax(m)
        #     ch=listc[r]
        #     text=text+ch
        #     cv2.putText(imgcopy,ch,(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,0,255),2)
     
    # custom_sent_tokenizer = PunktSentenceTokenizer(text) #A

    # tokenized = custom_sent_tokenizer.tokenize(text)
    # def process_content():
    #     try:
    #         for i in tokenized[:5]:
    #             words = nltk.word_tokenize(i)
    #             tagged = n  ltk.pos_tag(words)
    #             st.text(tagged)
    #     except Exception as e:
    #         st.text(str(e))
    # process_content()

    st.image(img_copy)
    # k=0
    # text1+=text[k]
    # for i in text:
    #     k+=1
    #     while k%3==0:
    #         text1+=text[k]
    #         k+=1
    # st.text(text1)
    t=['0']
    w=0
    for i in range(len(characters)):
        print(i)
    for i in range(len(boxes)):
        print(i)
    #     if 60>boxes[i][2]>0:
    #             t.append(character[i])
    #             w+=1
    # print(t)
                # st.text(text)
                    # ch=listc[k]
    # except Exception as e:
    #      st.error("Error")
