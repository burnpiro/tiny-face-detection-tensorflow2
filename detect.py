import cv2
import time
import numpy as np
import tensorflow as tf
from absl import app, flags
from absl.flags import FLAGS
from model.model import create_model
from config import cfg
from draw_boxes import draw_outputs

IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5
MAX_OUTPUT_SIZE = 49

flags.DEFINE_string('weights', './model-0.65.h5',
                    'path to weights file')
flags.DEFINE_string('image', './meme.jpg', 'path to input image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')


def main(_argv):
    model = create_model()
    model.load_weights(FLAGS.weights)

    proc_image = tf.keras.preprocessing.image.load_img(FLAGS.image,
                                                       target_size=(cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE))

    proc_image = tf.keras.preprocessing.image.img_to_array(proc_image)
    proc_image = np.expand_dims(proc_image, axis=0)
    proc_image - tf.keras.applications.mobilenet_v2.preprocess_input(proc_image)

    original_image = cv2.imread(FLAGS.image)

    t1 = time.time()
    pred = np.squeeze(model.predict(proc_image))
    t2 = time.time()
    processing_time = t2 - t1
    height, width, y_f, x_f, score = [a.flatten() for a in np.split(pred, pred.shape[-1], axis=-1)]

    coords = np.arange(pred.shape[0] * pred.shape[1])
    y = (y_f + coords // pred.shape[0]) / pred.shape[0]
    x = (x_f + coords % pred.shape[1]) / pred.shape[1]

    boxes = np.stack([y, x, height, width, score], axis=-1)
    boxes = boxes[np.where(boxes[..., -1] >= SCORE_THRESHOLD)]

    # does not work with TF GPU, uncomment only when using CPU
    # selected_indices = tf.image.non_max_suppression(boxes[..., :-1], boxes[..., -1], MAX_OUTPUT_SIZE, IOU_THRESHOLD)
    # selected_indices = tf.Session().run(selected_indices)

    # print(len(boxes))

    original_image = draw_outputs(original_image, boxes)
    original_image = cv2.putText(original_image, "Time: {:.2f}".format(processing_time), (0, 30),
                      cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    cv2.imwrite(FLAGS.output, original_image)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
