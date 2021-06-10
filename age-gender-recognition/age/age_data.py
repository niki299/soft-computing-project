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

    def set_up_folders(self):
        folder_names = ["1-5", "6-10", "11-15", "16-18", "19-21", "22-30", "31-44", "45-60", "61-80", "81-101"]
        for name in folder_names:
            full_train_path = '../../../UTKData/Training/' + name
            full_test_path = '../../../UTKData/Testing/' + name
            self.make_dirs([full_test_path, full_train_path])


    def get_dir_name_from_label(self, label_num):
        dir_name = ""
        if label_num <= 5:
            dir_name = "1-5"
        elif label_num <= 10:
            dir_name = "6-10"
        elif label_num <= 15:
            dir_name = "11-15"
        elif label_num <= 18:
            dir_name = "16-18"
        elif label_num <= 21:
            dir_name = "19-21"
        elif label_num <= 30:
            dir_name = "22-30"
        elif label_num <= 44:
            dir_name = "31-44"
        elif label_num <= 60:
            dir_name = "45-60"
        elif label_num <= 80:
            dir_name = "61-80"
        else:
            dir_name = "81-101"
        return dir_name

    def load(self, folder):
        files = list(self.get_files(folder))
        for (i, path) in enumerate(files):
            image = cv2.imread(path)
            fname = os.path.basename(path)
            label_num = int(fname.split('_')[0])
            image = resize_image(image)
            image = image2gray(image)
            dir_name = self.get_dir_name_from_label(label_num)

            if (i % 10) == 0:
                src = os.path.join('../../../UTKData/Testing/' + dir_name, fname)
                cv2.imwrite(src, image)
            else:
                src = os.path.join('../../../UTKData/Training/' + dir_name, fname)
                cv2.imwrite(src, image)


def image2gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_image(image, width=128, height=128):
    return cv2.resize(image, (width, height), cv2.INTER_AREA)


if __name__ == '__main__':
    id = ImageDataset()
    id.load("../../../UTKData/UTKFace")
