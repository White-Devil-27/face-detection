
import cv2
from random import randrange

# Default frontface detection pre-defined model of opencv
face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# img = cv2.imread('keerthy.png')

# Read a image and resize the image
src = cv2.imread('emma.png', cv2.IMREAD_UNCHANGED)

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
cv2.imshow("Face detection",img)

# convert_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# face_quartinates = face_data.detectMultiScale(convert_img)

# print(face_quartinates)

# for (x,y,w,h) in face_quartinates:
#     cv2.rectangle(img,(x,y),(x+w,y+w),(0,255,0),3)

# cv2.imshow("Face detection",img)

# Just press any key to exit the terminal
cv2.waitKey()

print("Its working")