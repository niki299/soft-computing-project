import os
from shutil import copyfile
import cv2
import numpy as np


class ImageDataset:
    def __init__(self):
        pass

    def get_files(self, folder):
        filenames = os.listdir(folder)
        for filename in filenames:
            filepath = os.path.join(folder, filename)
            yield filepath

    def make_dirs(self, paths):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    def load(self, folder):
        output_dir_train_male = '../../../UTKData/UTKFace/Training/male'
        output_dir_train_female = '../../../UTKData/UTKFace/Training/female'
        output_dir_test_male = '../../../UTKData/UTKFace/Testing/male'
        output_dir_test_female = '../../../UTKData/UTKFace/Testing/female'
        self.make_dirs( [output_dir_train_male, output_dir_train_female, output_dir_test_male, output_dir_test_female] )

        files = list(self.get_files(folder))
        for (i, path) in enumerate(files):
            image = cv2.imread(path)
            fname = os.path.basename(path)
            label = fname.split('_')[1]
            image = resize_image(image)
            image = image2gray(image)
            if (i % 10) == 0:
                if(label == "0"):
                    src = os.path.join(output_dir_test_male, fname)
                    cv2.imwrite(src, image)
                else:
                    src = os.path.join(output_dir_test_female, fname)
                    cv2.imwrite(src, image)
            else:
                if (label == "0"):
                    src = os.path.join(output_dir_train_male, fname)
                    cv2.imwrite(src, image)
                else:
                    src = os.path.join(output_dir_train_female, fname)
                    cv2.imwrite(src, image)

    def get_data(self):
        return np.array(self.images), np.array(self.labels)


def image2gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_image(image, width=128, height=128):
    return cv2.resize(image, (width, height), cv2.INTER_AREA)


if __name__ == '__main__':
    klasa = ImageDataset()
    klasa.load("../../../UTKData/UTKFace")
