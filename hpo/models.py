"""
Build mammogram classifiers with settings supplied by the search.

This module creates models trained from scratch or adapted from a pretrained
model, along with image variations and training settings for each candidate.
"""

from __future__ import annotations

import tensorflow as tf

from modules.config import Config

from .config import HPOConfig, TransferSearchSpec
from .regularized_config import RegularizedSearchSpec, SearchSpec


def create_augmentation(
    spec: SearchSpec, name: str = "hpo_training_only_augmentation"
) -> tf.keras.Sequential:
    """
    Create the candidate's random image variations to help reduce overfitting during
    training.
    """
    layers: list[tf.keras.layers.Layer] = [tf.keras.layers.RandomFlip("horizontal")]

    if spec.rotation > 0:
        layers.append(tf.keras.layers.RandomRotation(spec.rotation, fill_mode="reflect"))
    if spec.translation > 0:
        layers.append(
            tf.keras.layers.RandomTranslation(
                spec.translation, spec.translation, fill_mode="reflect"
            )
        )
    if spec.zoom > 0:
        layers.append(tf.keras.layers.RandomZoom(spec.zoom, spec.zoom, fill_mode="reflect"))
    if spec.contrast > 0:
        layers.append(tf.keras.layers.RandomContrast(spec.contrast))

    return tf.keras.Sequential(layers, name=name)


def compile_binary_model(
    model: tf.keras.Model,
    learning_rate: float,
    label_smoothing: float,
) -> None:
    """
    Prepare a candidate to learn benign versus malignant labels with its chosen training
    settings.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )


def build_transfer_model(config: HPOConfig, spec: TransferSearchSpec) -> tf.keras.Model:
    """
    Build a candidate classifier around a pretrained image model.

    Keep the pretrained features fixed initially and apply the candidate's choices
    for the new classification layers.
    """
    inputs = tf.keras.Input(
        (config.image_size, config.image_size, 3), name="image_0_255"
    )
    features = create_augmentation(spec)(inputs)

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(config.image_size, config.image_size, 3),
    )
    backbone.trainable = False
    features = backbone(features, training=False)

    if spec.pooling == "average":
        features = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(features)
    elif spec.pooling == "maximum":
        features = tf.keras.layers.GlobalMaxPooling2D(name="global_max_pool")(features)
    else:
        raise ValueError(f"Unknown pooling: {spec.pooling}")

    features = tf.keras.layers.Dropout(spec.head_dropout_1, name="head_dropout_1")(features)
    features = tf.keras.layers.Dense(
        spec.dense_units,
        activation=spec.head_activation,
        kernel_regularizer=tf.keras.regularizers.l2(spec.l2_strength),
        name="classification_features",
    )(features)
    features = tf.keras.layers.Dropout(spec.head_dropout_2, name="head_dropout_2")(features)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        dtype="float32",
        name="malignancy_probability",
    )(features)

    model = tf.keras.Model(inputs, outputs, name="efficientnetb0_transfer_hpo")
    compile_binary_model(model, spec.head_learning_rate, spec.label_smoothing)

    return model


def enable_fine_tuning(model: tf.keras.Model, spec: TransferSearchSpec) -> None:
    """
    Allow selected later layers of the pretrained model to adapt using the candidate's
    training settings.
    """
    backbones = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower()
    ]
    if len(backbones) != 1:
        raise RuntimeError(f"Invalid backbone count: {len(backbones)}")

    backbone = backbones[0]
    backbone.trainable = True

    cutoff = max(0, len(backbone.layers) - spec.fine_tune_last_layers)

    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= cutoff and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )

    compile_binary_model(model, spec.fine_tune_learning_rate, spec.label_smoothing)


def _regularized_convolution_blocks(
    features: tf.Tensor, spec: RegularizedSearchSpec, weight_regularizer
) -> tf.Tensor:
    """
    Build the image-feature layers with the candidate's size and overfitting controls.
    """
    filters_by_block = tuple(
        min(512, spec.initial_filters * (2 ** block_index))
        for block_index in range(spec.block_count)
    )
    spatial_dropout_start = max(1, len(filters_by_block) - 1)

    for block_index, filters in enumerate(filters_by_block, start=1):
        features = tf.keras.layers.Conv2D(
            filters,
            3,
            padding="same",
            use_bias=not spec.use_batch_norm,
            kernel_initializer="he_normal",
            kernel_regularizer=weight_regularizer,
            name=f"block{block_index}_conv",
        )(features)

        if spec.use_batch_norm:
            features = tf.keras.layers.BatchNormalization(
                name=f"block{block_index}_bn"
            )(features)

        features = tf.keras.layers.Activation(
            spec.activation, name=f"block{block_index}_{spec.activation}"
        )(features)
        features = tf.keras.layers.MaxPooling2D(
            name=f"block{block_index}_pool"
        )(features)

        if block_index >= spatial_dropout_start and spec.spatial_dropout > 0:
            features = tf.keras.layers.SpatialDropout2D(
                spec.spatial_dropout, name=f"block{block_index}_drop"
            )(features)

    return features


def build_regularized_model(
    config: Config, spec: RegularizedSearchSpec
) -> tf.keras.Model:
    """
    Build a classifier that learns from scratch using the candidate's model and training
    choices.
    """
    inputs = tf.keras.Input(
        (config.image_size, config.image_size, 3), name="image_0_255"
    )
    features = create_augmentation(
        spec, name="regularized_training_only_augmentation"
    )(inputs)
    features = tf.keras.layers.Rescaling(
        1.0 / 255.0, name="cnn_rescale_0_1"
    )(features)

    weight_regularizer = (
        tf.keras.regularizers.l2(spec.l2_strength)
        if spec.l2_strength > 0
        else None
    )
    features = _regularized_convolution_blocks(features, spec, weight_regularizer)

    if spec.pooling == "maximum":
        features = tf.keras.layers.GlobalMaxPooling2D(name="global_max_pool")(features)
    else:
        features = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(features)

    features = tf.keras.layers.Dense(
        spec.dense_units,
        activation=spec.activation,
        kernel_regularizer=weight_regularizer,
        name="classification_features",
    )(features)

    if spec.head_dropout > 0:
        features = tf.keras.layers.Dropout(
            spec.head_dropout, name="head_dropout"
        )(features)

    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid", name="malignancy_probability"
    )(features)
    model = tf.keras.Model(inputs, outputs, name="regularized_cnn")
    compile_binary_model(model, spec.learning_rate, spec.label_smoothing)

    return model
