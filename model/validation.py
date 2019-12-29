import os
import numpy as np
import tensorflow as tf
from config import cfg


class Validation(tf.keras.callbacks.Callback):
    def get_box_highets_percentage(self, mask):
        reshaped = mask.reshape(mask.shape[0], np.prod(mask.shape[1:-1]), -1)

        score_ind = np.argmax(reshaped[..., -1], axis=-1)
        unraveled = np.array(np.unravel_index(score_ind, mask.shape[:-1])).T[:, 1:]

        cell_y, cell_x = unraveled[..., 0], unraveled[..., 1]
        boxes = mask[np.arange(mask.shape[0]), cell_y, cell_x]

        h, w, offset_y, offset_x = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]

        return np.stack([cell_y + offset_y, cell_x + offset_x,
                         cfg.NN.GRID_SIZE * h, cfg.NN.GRID_SIZE * w], axis=-1)

    def __init__(self, generator):
        self.generator = generator

    def on_epoch_end(self, epoch, logs):
        mse = 0
        intersections = 0
        unions = 0

        for i in range(len(self.generator)):
            batch_images, gt = self.generator[i]
            pred = self.model.predict_on_batch(batch_images)

            pred = self.get_box_highets_percentage(pred)
            gt = self.get_box_highets_percentage(gt)

            mse += np.linalg.norm(gt - pred, ord='fro') / pred.shape[0]

            pred = np.maximum(pred, 0)
            gt = np.maximum(gt, 0)

            diff_height = np.minimum(gt[:,0] + gt[:,2], pred[:,0] + pred[:,2]) - np.maximum(gt[:,0], pred[:,0])
            diff_width = np.minimum(gt[:,1] + gt[:,3], pred[:,1] + pred[:,3]) - np.maximum(gt[:,1], pred[:,1])
            intersection = np.maximum(diff_width, 0) * np.maximum(diff_height, 0)

            area_gt = gt[:,2] * gt[:,3]
            area_pred = pred[:,2] * pred[:,3]
            union = np.maximum(area_gt + area_pred - intersection, 0)

            intersections += np.sum(intersection * (union > 0))
            unions += np.sum(union)

        iou = np.round(intersections / (unions + tf.keras.backend.epsilon()), 4)
        logs["val_iou"] = iou

        mse = np.round(mse, 4)
        logs["val_mse"] = mse

        print(" - val_iou: {} - val_mse: {}\n".format(iou, mse))
