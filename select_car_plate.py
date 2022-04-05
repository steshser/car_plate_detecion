import cv2 as cv
INCOME_PLATE_DATA = 'haarcascade_russian_plate_number.xml'


def select_car_plate(video_capture):
    success, img = video_capture.read()
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    plate = cv.CascadeClassifier(INCOME_PLATE_DATA)
    results = plate.detectMultiScale(gray_img, scaleFactor=3.5, minNeighbors=2)
    for (x, y, w, h) in results:
        cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
    return img
