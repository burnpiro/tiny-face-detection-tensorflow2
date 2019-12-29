import os
import sys
import math
import numpy as np
import tensorflow as tf
from config import cfg


# Input: [x0, y0, w, h, blur, expression, illumination, invalid, occlusion, pose]
# Output: x0, y0, w, h
def get_box(data):
    x0 = int(data[0])
    y0 = int(data[1])
    w = int(data[2])
    h = int(data[3])
    return x0, y0, w, h


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, file_path, config_path, debug=False):
        self.boxes = []
        self.debug = debug
        self.data_path = file_path

        if not os.path.isfile(config_path):
            print("File path {} does not exist. Exiting...".format(config_path))
            sys.exit()

        if not os.path.isdir(file_path):
            print("Images folder path {} does not exist. Exiting...".format(file_path))
            sys.exit()

        with open(config_path) as fp:
            line = fp.readline()
            cnt = 1
            while line:
                num_of_obj = int(fp.readline())
                for i in range(num_of_obj):
                    obj_box = fp.readline().split(' ')
                    x0, y0, w, h = get_box(obj_box)
                    if w == 0:
                        # remove boxes with no width
                        continue
                    if h == 0:
                        # remove boxes with no height
                        continue
                    # Because our network is outputting 7x7 grid then it's not worth processing images with more than
                    # 5 faces because it's highly probable they are close to each other.
                    # You could remove this filter if you decide to switch to larger grid (like 14x14)
                    # Don't worry about number of train data because even with this filter we have around 16k samples
                    if num_of_obj > 5:
                        continue
                    self.boxes.append((line.strip(), x0, y0, w, h))
                if num_of_obj == 0:
                    obj_box = fp.readline().split(' ')
                    x0, y0, w, h = get_box(obj_box)
                    self.boxes.append((line.strip(), x0, y0, w, h))
                line = fp.readline()
                cnt += 1

    def __len__(self):
        return math.ceil(len(self.boxes) / cfg.TRAIN.BATCH_SIZE)

    def __getitem__(self, idx):
        boxes = self.boxes[idx * cfg.TRAIN.BATCH_SIZE:(idx + 1) * cfg.TRAIN.BATCH_SIZE]

        batch_images = np.zeros((len(boxes), cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE, 3), dtype=np.float32)
        batch_boxes = np.zeros((len(boxes), cfg.NN.GRID_SIZE, cfg.NN.GRID_SIZE, 5), dtype=np.float32)

        for i, row in enumerate(boxes):
            path, x0, y0, w, h = row

            proc_image = tf.keras.preprocessing.image.load_img(self.data_path + path)

            image_width = proc_image.width
            image_height = proc_image.height

            proc_image = tf.keras.preprocessing.image.load_img(self.data_path + path,
                                                               target_size=(cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE))

            proc_image = tf.keras.preprocessing.image.img_to_array(proc_image)
            proc_image = np.expand_dims(proc_image, axis=0)
            proc_image - tf.keras.applications.mobilenet_v2.preprocess_input(proc_image)

            batch_images[i] = proc_image

            # make sure none of the points is out of image border
            x0 = max(x0, 0)
            y0 = max(y0, 0)

            x0 = min(x0, image_width)
            y0 = min(y0, image_height)

            x_c = (cfg.NN.GRID_SIZE / image_width) * x0
            y_c = (cfg.NN.GRID_SIZE / image_height) * y0

            floor_y = math.floor(y_c)  # handle case when x i on the corner
            floor_x = math.floor(x_c)  # handle case when y i on the corner

            if floor_x == cfg.NN.GRID_SIZE and self.debug:
                print(path, x0, y0, w, h, x_c)
            if floor_y == cfg.NN.GRID_SIZE and self.debug:
                print(path, x0, y0, w, h, y_c)

            batch_boxes[i, floor_y, floor_x, 0] = h / image_height
            batch_boxes[i, floor_y, floor_x, 1] = w / image_width
            batch_boxes[i, floor_y, floor_x, 2] = y_c - floor_y
            batch_boxes[i, floor_y, floor_x, 3] = x_c - floor_x
            batch_boxes[i, floor_y, floor_x, 4] = 1

        return batch_images, batch_boxes
