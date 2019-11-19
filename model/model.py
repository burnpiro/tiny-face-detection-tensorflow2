import tensorflow as tf
from config import cfg


def create_model(trainable=False):
    base = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE, 3),
                                                          alpha=cfg.NN.ALPHA, weights='imagenet', include_top=False)

    for layer in base.layers:
        layer.trainable = trainable

    out = base.get_layer('block_16_project_BN').output
    # Change 112 to whatever is the size of block_16_project_BN, "112" value is correct for 0.35 ALPHA, 448 is for 1.4
    # Depends on your output complexity you might want to add another Conv2D layers (like one commented out displayed below)
    out = tf.keras.layers.Conv2D(240, padding="same", kernel_size=3, strides=1, activation="relu")(out)
    # out = tf.keras.layers.Conv2D(240, padding="same", kernel_size=3, strides=1, use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation('relu')(out)

    out = tf.keras.layers.Conv2D(5, padding='same', kernel_size=1, activation='sigmoid')(out)

    model = tf.keras.Model(inputs=base.input, outputs=out)

    # divide by 2 since d/dweight learning_rate * weight^2 = 2 * learning_rate * weight
    # see https://arxiv.org/pdf/1711.05101.pdf
    regularizer = tf.keras.regularizers.l2(cfg.TRAIN.WEIGHT_DECAY / 2)

    for weight in model.trainable_weights:
        with tf.keras.backend.name_scope('weight_regularizer'):
            model.add_loss(lambda: regularizer(weight))

    return model
