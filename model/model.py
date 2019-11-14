import tensorflow as tf
from config import cfg


def create_model(trainable=False):
    base = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE, 3),
                                                          alpha=cfg.NN.ALPHA, weights='imagenet', include_top=False)

    for layer in base.layers:
        layer.trainable = trainable

    block = base.get_layer('block_16_project_BN').output
    # UpSample base on the number of cells in the grid
    x = tf.keras.layers.UpSampling2D()(block)
    # Change 112 to whatever is the size of block_16_project_BN, "112" value is correct for 0.35 ALPHA, 448 is for 1.4
    x = tf.keras.layers.Conv2D(448, padding="same", kernel_size=3, strides=1, activation="relu")(x)
    x = tf.keras.layers.Conv2D(448, padding="same", kernel_size=3, strides=1, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(5, padding='same', kernel_size=1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=base.input, outputs=x)

    # divide by 2 since d/dweight learning_rate * weight^2 = 2 * learning_rate * weight
    # see https://arxiv.org/pdf/1711.05101.pdf
    regularizer = tf.keras.regularizers.l2(cfg.TRAIN.WEIGHT_DECAY / 2)

    for weight in model.trainable_weights:
        with tf.keras.backend.name_scope('weight_regularizer'):
            model.add_loss(lambda: regularizer(weight))

    return model
