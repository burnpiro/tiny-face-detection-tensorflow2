import tensorflow as tf
from config import cfg


def detect_loss():
    def get_box_highest_percentage(arr):
        shape = tf.shape(arr)

        reshaped = tf.reshape(arr, (shape[0], tf.reduce_prod(shape[1:-1]), -1))

        # returns array containing the index of the highest percentage of each batch
        # where 0 <= index <= height * width
        max_prob_ind = tf.argmax(reshaped[..., -1], axis=-1, output_type=tf.int32)

        # turn indices (batch, y * x) into (batch, y, x)
        # returns (3, batch) tensor
        unraveled = tf.unravel_index(max_prob_ind, shape[:-1])

        # turn tensor into (batch, 3) and keep only (y, x)
        unraveled = tf.transpose(unraveled)[:, 1:]
        y, x = unraveled[..., 0], unraveled[..., 1]

        # stack indices and create (batch, 5) tensor which
        # contains height, width, offset_y, offset_x, percentage
        indices = tf.stack([tf.range(shape[0]), y, x], axis=-1)
        box = tf.gather_nd(arr, indices)

        y, x = tf.cast(y, tf.float32), tf.cast(x, tf.float32)

        # transform box to (y + offset_y, x + offset_x, GRID_SIZE * height, GRID_SIZE * width, obj)
        # output is (batch, 5)
        out = tf.stack([y + box[..., 2], x + box[..., 3],
                        cfg.NN.GRID_SIZE * box[..., 0], cfg.NN.GRID_SIZE * box[..., 1],
                        box[..., -1]], axis=-1)

        return out

    def loss(y_true, y_pred):
        # get the box with the highest percentage in each image
        true_box = get_box_highest_percentage(y_true)
        pred_box = get_box_highest_percentage(y_pred)

        # object loss
        obj_loss = tf.keras.losses.binary_crossentropy(y_true[..., 4:5], y_pred[..., 4:5])

        # mse with the boxes that have the highest percentage
        box_loss = tf.reduce_sum(tf.math.squared_difference(true_box[..., :-1], pred_box[..., :-1]))

        return tf.reduce_sum(obj_loss) + box_loss

    return loss
