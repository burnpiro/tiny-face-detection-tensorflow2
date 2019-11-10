import os
import sys
import tensorflow as tf


def load_text_file(file='./data/wider_face_split/wider_face_train_bbx_gt.txt'):
    if not os.path.isfile(file):
       print("File path {} does not exist. Exiting...".format(file))
       sys.exit()

    images_path = []
    labels = []

    with open(file) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            images_path.append(line.strip())
            num_of_obj = int(fp.readline())
            temp_labels = []
            for i in range(num_of_obj):
                obj_box = fp.readline()
                temp_labels.append(obj_box.split(' ')[:4])
            if num_of_obj == 0:
                obj_box = fp.readline()
                temp_labels.append(obj_box.split(' ')[:4])
            labels.append(temp_labels)
            line = fp.readline()
            cnt += 1

    return images_path, labels


images_path, labels = load_text_file()

print(images_path[2])
print(labels[2])
