import argparse

import cv2
import mahotas
from sklearn.externals import joblib

import dataset
from hog import HOG

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-m', '--model', required=True, help='Path to where the model is stored.')
argument_parser.add_argument('-i', '--image', required=True, help='Path to the image file.')
arguments = vars(argument_parser.parse_args())

model = joblib.load(arguments['model'])

hog = HOG(orientations=18, pixels_per_cell=(10, 10), cells_per_block=(1, 1), transform=True)

image = cv2.imread(arguments['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)
(contours, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted([(c, cv2.boundingRect(c)[0]) for c in contours], key=lambda x: x[1])

for (c, _) in contours:
    (x, y, w, h) = cv2.boundingRect(c)

    if w >= 7 and h >= 20:
        roi = gray[y:y + h, x:x + w]
        thresh = roi.copy()
        T = mahotas.thresholding.otsu(roi)
        thresh[thresh > T] = 255
        thresh = cv2.bitwise_not(thresh)

        thresh = dataset.deskew(thresh, 20)
        thresh = dataset.center_extent(thresh, (20, 20))

        cv2.imshow('thresh', thresh)

        histogram = hog.describe(thresh)
        digit = model.predict([histogram])[0]
        print('I think that number is: {}'.format(digit))

        cv2.rectangle(image, (x, y), (x + w, y + h), (128, 0, 128), 1)
        cv2.putText(image, str(digit), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (128, 0, 128), 2)
        cv2.imshow('image', image)
        cv2.waitKey(0)
