from flask import Flask, render_template,request
import cv2
from random import randrange
from io import BytesIO
from PIL import Image
import base64
import re

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload',methods=['POST'])
def upload(): 

    image = request.files['logo'].read()
    # print(type(image))
    
    image_PIL = Image.open(BytesIO(image))
    image_PIL.save("image.png")

    # Default frontface detection pre-defined model of opencv
    face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Read a image and resize the image
    src = cv2.imread("image.png", cv2.IMREAD_UNCHANGED)

    # Percent of which image is resized
    scale_percent = 50

    # Calculate the 50 percent of original dimensions
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    # Resize image and its parameters
    img = cv2.resize(src, dsize,interpolation = cv2.INTER_AREA)

    # Convert the image to black and white
    convert_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Detect face
    face_quartinates = face_data.detectMultiScale(convert_img)

    # Draw the rectangle shape using detected face-quartinates
    for (x,y,w,h) in face_quartinates:
        cv2.rectangle(img,(x,y),(x+w,y+w),(randrange(255),randrange(255),randrange(255)),3)

    # Display the image
    # cv2.imshow("Face detection",img)
    cv2.imwrite('static/capture.png',img)

    # convert_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # face_quartinates = face_data.detectMultiScale(convert_img)

    # print(face_quartinates)

    # for (x,y,w,h) in face_quartinates:
    #     cv2.rectangle(img,(x,y),(x+w,y+w),(0,255,0),3)

    # cv2.imshow("Face detection",img)

    # Just press any key to exit the terminal
    # cv2.waitKey()

    # print("Its working")
    return render_template('index.html',image='capture.png')