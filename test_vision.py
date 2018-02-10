import numpy as np
import vision
import unittest
from vision import process
import cv2
import cscore
import csv


def test_black():
    output = vision.process(np.zeros(shape=(320, 240, 3), dtype=np.uint8))
    assert output == []


def test_sample_images():
    with open('sample_img/tests.csv', 'r') as csvfile:
        # filename, x, y, w, h
        testreader = csv.reader(csvfile, delimiter=',')
        for sample in testreader:
            image = cv2.imread('sample_img/' + sample[0])
            # Rescale if necessary
            height, width, channels = image.shape
            angle, position, contour_area = vision.process(image)
            num_cube = len(process(image)) -1
            assert sample[1] == num_cube