

import json
import codecs

import numpy as np
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO

import cv2
import time

import json


if __name__ == '__main__':

    address = './face_detection.json'

    jsonData = []

    with codecs.open(address, 'rU', 'utf-8') as js:
        for line in js:
            jsonData.append(json.loads(line))

    print(f"{len(jsonData)} image found!")

    print("Sample row:")

    jsonData[0]


    # load images from url and save into images

    images = []

    for data in tqdm(jsonData):
        response = requests.get(data['content'])
        img = np.asarray(Image.open(BytesIO(response.content)))
        images.append([img, data["annotation"]])


    #Writing images

    count = 1

    totalfaces = 0

    start = time.time()

    for image in images:
        img = image[0]
        cv2.imwrite('./face-detection-images/face_image_{}.jpg'.format(count), img)
        jsonData[count - 1]['content'] = 'face_image_{}.jpg'.format(count)
        count += 1

    with open('./face-detection-images/result.json', 'w') as fp:
        json.dump(jsonData, fp)


    end = time.time()

    print("Total test images with faces : {}".format(len(images)))
    print("Sucessfully tested {} images".format(count - 1))
    print("Execution time in seconds {}".format(end - start))
    print("Total Faces Detected {}".format(totalfaces))