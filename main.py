import cv2 as cv
from select_car_plate import select_car_plate
import numpy as np
import easyocr
import imutils
from matplotlib import pyplot as pl

# download video from https://disk.yandex.ru/i/eyo02g8TmX8qPg or try use another video file
VIDEO_PATH = 'videos/Traffic Flow In The Highway.mp4'
video_capture = cv.VideoCapture(VIDEO_PATH)

while True:
    img = select_car_plate(video_capture)
    cv.imshow('Result', cv.resize(img, (800, 600)))
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

