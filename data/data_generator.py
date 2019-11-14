import os
import sys
import math
import numpy as np
import tensorflow as tf
from config import cfg


# Input: [x0, y0, w, h]
# Output: x0, y0, x1, y1
def get_box(data):
    x0 = int(data[0])
    y0 = int(data[1])
    x1 = int(data[0]) + int(data[2])
    y1 = int(data[1]) + int(data[3])
    return x0, y0, x1, y1


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, file_path, config_path, rnd_multiply=True, rnd_color=True, rnd_crop=True, rnd_flip=False,
                 debug=False):
        self.boxes = []
        self.debug = debug
        self.data_path = file_path

        if not os.path.isfile(config_path):
            print("File path {} does not exist. Exiting...".format(config_path))
            sys.exit()

        if not os.path.isdir(file_path):
            print("Folder path {} does not exist. Exiting...".format(file_path))
            sys.exit()

        with open(config_path) as fp:
            line = fp.readline()
            cnt = 1
            while line:
                num_of_obj = int(fp.readline())
                for i in range(num_of_obj):
                    obj_box = fp.readline().split(' ')
                    x0, y0, x1, y1 = get_box(obj_box)
                    if x0 >= x1:
                        continue
                    if y0 >= y1:
                        continue
                    self.boxes.append((line.strip(), x0, y0, x1, y1))
                if num_of_obj == 0:
                    obj_box = fp.readline().split(' ')
                    x0, y0, x1, y1 = get_box(obj_box)
                    self.boxes.append((line.strip(), x0, y0, x1, y1))
                line = fp.readline()
                cnt += 1

    def __len__(self):
        return math.ceil(len(self.boxes) / cfg.TRAIN.BATCH_SIZE)

    def __getitem__(self, idx):
        boxes = self.boxes[idx * cfg.TRAIN.BATCH_SIZE:(idx + 1) * cfg.TRAIN.BATCH_SIZE]

        batch_images = np.zeros((len(boxes), cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE, 3), dtype=np.float32)
        batch_boxes = np.zeros((len(boxes), cfg.NN.GRID_SIZE, cfg.NN.GRID_SIZE, 5), dtype=np.float32)

        for i, row in enumerate(boxes):
            path, x0, y0, x1, y1 = row

            proc_image = tf.keras.preprocessing.image.load_img(self.data_path + path)

            image_width = proc_image.width
            image_height = proc_image.height

            proc_image = tf.keras.preprocessing.image.load_img(self.data_path + path, target_size=(cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE))

            proc_image = tf.keras.preprocessing.image.img_to_array(proc_image)
            proc_image = np.expand_dims(proc_image, axis=0)
            proc_image - tf.keras.applications.mobilenet_v2.preprocess_input(proc_image)

            batch_images[i] = proc_image

            # make sure none of the points is out of image border
            tmp = x0
            x0 = min(x0, x1)
            x1 = max(tmp, x1)

            tmp = y0
            y0 = min(y0, y1)
            y1 = max(tmp, y1)

            x0 = max(x0, 0)
            y0 = max(y0, 0)

            y0 = min(y0, image_height)
            x0 = min(x0, image_width)
            y1 = min(y1, image_height)
            x1 = min(x1, image_width)

            # some of the data is invalid and goes over image boundaries
            # because of that both x0 and x1 are set to be max height
            if x1 == x0:
                x0 = x0 - 1
            if y1 == y0:
                y0 = y0 - 1

            x_c = (cfg.NN.GRID_SIZE / image_width) * (x0 + (x1 - x0) / 2)
            y_c = (cfg.NN.GRID_SIZE / image_height) * (y0 + (y1 - y0) / 2)

            floor_y = math.floor(y_c)  # handle case when x i on the corner
            floor_x = math.floor(x_c)  # handle case when y i on the corner

            if floor_x == cfg.NN.GRID_SIZE:
                print(path, x0, y0, x1, y1, x_c)
            if floor_y == cfg.NN.GRID_SIZE:
                print(path, x0, y0, x1, y1, y_c)

            batch_boxes[i, floor_y, floor_x, 0] = (y1 - y0) / image_height
            batch_boxes[i, floor_y, floor_x, 1] = (x1 - x0) / image_width
            batch_boxes[i, floor_y, floor_x, 2] = y_c - floor_y
            batch_boxes[i, floor_y, floor_x, 3] = x_c - floor_x
            batch_boxes[i, floor_y, floor_x, 4] = 1

        return batch_images, batch_boxes
