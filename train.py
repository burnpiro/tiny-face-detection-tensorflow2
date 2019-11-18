import tensorflow as tf
from datetime import datetime
from model.model import create_model
from model.validation import Validation
from data.data_generator import DataGenerator
from model.loss import detect_loss
from config import cfg

TRAINABLE = False


def main():
    model = create_model(trainable=TRAINABLE)

    # if TRAINABLE:
    #     model.load_weights(WEIGHTS)

    train_datagen = DataGenerator(file_path=cfg.TRAIN.DATA_PATH, config_path=cfg.TRAIN.ANNOTATION_PATH)

    val_generator = DataGenerator(file_path=cfg.TEST.DATA_PATH, config_path=cfg.TEST.ANNOTATION_PATH, debug=False)
    validation_datagen = Validation(generator=val_generator)

    learning_rate = cfg.TRAIN.LEARNING_RATE
    if TRAINABLE:
        learning_rate /= 10

    optimizer = tf.keras.optimizers.SGD(lr=learning_rate, decay=cfg.TRAIN.LR_DECAY, momentum=0.9, nesterov=False)
    model.compile(loss=detect_loss(), optimizer=optimizer, metrics=[])

    checkpoint = tf.keras.callbacks.ModelCheckpoint("model-{val_iou:.2f}.h5", monitor="val_iou", verbose=1,
                                                    save_best_only=True,
                                                    save_weights_only=True, mode="max")
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_iou", patience=cfg.TRAIN.PATIENCE, mode="max")
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_iou", factor=0.6, patience=5, min_lr=1e-6, verbose=1,
                                                     mode="max")

    # Define the Keras TensorBoard callback.
    logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    model.fit_generator(generator=train_datagen,
                        epochs=cfg.TRAIN.EPOCHS,
                        callbacks=[tensorboard_callback, validation_datagen, checkpoint, reduce_lr, stop],
                        shuffle=True,
                        verbose=1)


if __name__ == "__main__":
    main()
