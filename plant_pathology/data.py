from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf

tf.data.experimental.enable_debug_mode()


class DatasetHandler:
    def __init__(self, data_csv, images_path):
        df = self._read_csv(data_csv, images_path)
        images = df["image"].values
        labels = df["label"].str.split().values
        self.binarizer = MultiLabelBinarizer()
        labels_binarized = self.binarizer.fit_transform(labels)
        self.train_ds, self.val_ds = self._create_datasets(
            images, labels_binarized
        )

    @staticmethod
    def _read_csv(data_csv, images_path):
        df = pd.read_csv(data_csv)
        df["image"] = df["image"].apply(
            lambda image_filename: str(Path(images_path) / image_filename)
        )
        df = df.rename(columns={"labels": "label"})
        return df

    @staticmethod
    def _create_datasets(images, labels):
        x_train, x_val, y_train, y_val = train_test_split(
            images,
            labels,
            test_size=0.2,
        )
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        return train_ds, val_ds


class ImageReader:
    def __init__(self, image_size, interpolation, crop_to_aspect_ratio):
        self.image_size = image_size
        self.interpolation = interpolation
        self.crop_to_aspect_ratio = crop_to_aspect_ratio

    @staticmethod
    def load_image(path, num_channels):
        """Load an image from a path and resize it."""
        img = tf.io.read_file(path)
        img = tf.image.decode_image(
            img, channels=num_channels, expand_animations=False
        )
        return img

    @staticmethod
    def crop(image, image_size, interpolation, crop_to_aspect_ratio=False):
        height, width, num_channels = image_size
        if crop_to_aspect_ratio:
            image = tf.keras.preprocessing.image.smart_resize(
                image, image_size, interpolation=interpolation
            )
        else:
            image = tf.image.resize(
                image, (height, width), method=interpolation
            )
        image.set_shape((image_size[0], image_size[1], num_channels))
        return image

    def __call__(self, image_path):
        image = self.load_image(image_path, self.image_size[-1])
        image = self.crop(
            image,
            self.image_size,
            self.interpolation,
            self.crop_to_aspect_ratio,
        )
        image /= 255
        return image
