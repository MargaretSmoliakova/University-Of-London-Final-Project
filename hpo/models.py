from __future__ import annotations

import tensorflow as tf

from .config import HPOConfig, TransferSearchSpec


def create_augmentation(spec: TransferSearchSpec) -> tf.keras.Sequential:
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

    return tf.keras.Sequential(layers, name="hpo_training_only_augmentation")


def compile_transfer_model(
    model: tf.keras.Model,
    learning_rate: float,
    label_smoothing: float,
) -> None:
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
    compile_transfer_model(model, spec.head_learning_rate, spec.label_smoothing)

    return model


def enable_fine_tuning(model: tf.keras.Model, spec: TransferSearchSpec) -> None:
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

    compile_transfer_model(model, spec.fine_tune_learning_rate, spec.label_smoothing)
