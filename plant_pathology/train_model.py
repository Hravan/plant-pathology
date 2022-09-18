import argparse
from functools import partial
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import compute_class_weight
import tensorflow as tf

from plant_pathology.confusion_matrix import ConfusionMatrixLogger
from plant_pathology.data import DatasetHandler, ImageReader
from plant_pathology import nets


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("images_path")
    arg_parser.add_argument("annotations_path")
    arg_parser.add_argument("batch_size", type=int)
    args = arg_parser.parse_args()

    train_model(args.images_path, args.annotations_path, args.batch_size)


def train_model(images_path, annotations_csv_path, batch_size):
    dataset = DatasetHandler(annotations_csv_path, images_path)

    image_reader = ImageReader((200, 300, 3), "bilinear", False)

    train_ds = (
        dataset.train_ds.shuffle(len(dataset.train_ds))
        .map(lambda x, y: (image_reader(x), y))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = dataset.val_ds.map(lambda x, y: (image_reader(x), y)).batch(
        batch_size
    )

    model = nets.create_model((200, 300, 3), n_outputs=6)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["categorical_accuracy"],
        optimizer="adam",
    )

    # Callbacks
    log_dir = pathlib.Path("logs/1-baseline")
    tb_callback = tf.keras.callbacks.TensorBoard(str(log_dir))
    # cm_callback = tf.keras.callbacks.LambdaCallback(
    #    on_epoch_end=ConfusionMatrixLogger(
    #        model, val_ds, label_encoder, log_dir
    #    )
    # )
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(log_dir / "model")
    callbacks = [tb_callback, checkpoint_callback]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=300,
        callbacks=callbacks,
        # class_weight=dict(enumerate(class_weights)),
    )

    pred = model.predict(val_ds)
    pred = pred.argmax(-1)
    y = [y.numpy() for _, y_batch in val_ds for y in y_batch]
    print(f'F1 score: {f1_score(y, pred, average="macro")}')
    print(confusion_matrix(y, pred))


if __name__ == "__main__":
    main()
