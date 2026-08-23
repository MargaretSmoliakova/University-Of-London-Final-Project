from __future__ import annotations

from dataclasses import asdict, dataclass

import tensorflow as tf

from .config import Config


@dataclass(frozen=True)
class RegularizationSpec:
    name: str = "primary_regularized"
    l2_strength: float = 1e-4
    spatial_dropout: float = 0.10
    head_dropout: float = 0.40
    learning_rate: float = 3e-4
    use_augmentation: bool = True
    use_batch_norm: bool = True

    def serialisable(self) -> dict[str, object]:
        return asdict(self)


PRIMARY_REGULARIZATION = RegularizationSpec()


def augmentation_layers() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.025, fill_mode="reflect"),
            tf.keras.layers.RandomTranslation(0.04, 0.04, fill_mode="reflect"),
            tf.keras.layers.RandomZoom(0.08, 0.08, fill_mode="reflect"),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="training_only_augmentation",
    )


def compile_binary_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )


def build_scratch_cnn(
    config: Config,
    regularized: bool,
    specification: RegularizationSpec = PRIMARY_REGULARIZATION,
) -> tf.keras.Model:
    inputs = tf.keras.Input((config.image_size, config.image_size, 3), name="image_0_255")
    features = inputs
    if regularized and specification.use_augmentation:
        features = augmentation_layers()(features)
    features = tf.keras.layers.Rescaling(1.0 / 255.0, name="cnn_rescale_0_1")(features)
    weight_regularizer = (
        tf.keras.regularizers.l2(specification.l2_strength)
        if regularized and specification.l2_strength > 0
        else None
    )
    for block_index, filters in enumerate((32, 64, 128, 256), start=1):
        features = tf.keras.layers.Conv2D(
            filters,
            3,
            padding="same",
            use_bias=not (regularized and specification.use_batch_norm),
            kernel_initializer="he_normal",
            kernel_regularizer=weight_regularizer,
            name=f"block{block_index}_conv",
        )(features)
        if regularized and specification.use_batch_norm:
            features = tf.keras.layers.BatchNormalization(
                name=f"block{block_index}_bn"
            )(features)
        features = tf.keras.layers.Activation(
            "relu", name=f"block{block_index}_relu"
        )(features)
        features = tf.keras.layers.MaxPooling2D(
            name=f"block{block_index}_pool"
        )(features)
        if regularized and block_index >= 3 and specification.spatial_dropout > 0:
            features = tf.keras.layers.SpatialDropout2D(
                specification.spatial_dropout, name=f"block{block_index}_drop"
            )(features)
    features = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(features)
    features = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=weight_regularizer,
        name="classification_features",
    )(features)
    if regularized and specification.head_dropout > 0:
        features = tf.keras.layers.Dropout(
            specification.head_dropout, name="head_dropout"
        )(features)
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid", name="malignancy_probability"
    )(features)
    model = tf.keras.Model(
        inputs,
        outputs,
        name="regularized_cnn" if regularized else "baseline_cnn",
    )
    compile_binary_model(
        model,
        learning_rate=specification.learning_rate if regularized else 1e-3,
    )
    return model


def build_transfer_model(config: Config) -> tf.keras.Model:
    inputs = tf.keras.Input((config.image_size, config.image_size, 3), name="image_0_255")
    features = augmentation_layers()(inputs)
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(config.image_size, config.image_size, 3),
    )
    backbone.trainable = False
    features = backbone(features, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(features)
    features = tf.keras.layers.Dropout(0.40, name="head_dropout_1")(features)
    features = tf.keras.layers.Dense(
        128,
        activation="swish",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="classification_features",
    )(features)
    features = tf.keras.layers.Dropout(0.25, name="head_dropout_2")(features)
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid", name="malignancy_probability"
    )(features)
    model = tf.keras.Model(inputs, outputs, name="efficientnetb0_transfer")
    compile_binary_model(model, learning_rate=1e-3)
    return model


def enable_transfer_fine_tuning(model: tf.keras.Model, last_layers: int) -> None:
    backbones = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower()
    ]
    if len(backbones) != 1:
        raise RuntimeError(f"Invalid backbone count: {len(backbones)}")
    backbone = backbones[0]
    backbone.trainable = True
    cutoff = max(0, len(backbone.layers) - last_layers)
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= cutoff and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )
    compile_binary_model(model, learning_rate=1e-5)


def balanced_class_weights(metadata) -> dict[int, float]:
    counts = metadata["label"].value_counts().sort_index()
    if set(counts.index) != {0, 1}:
        raise ValueError("Missing training class")
    total = float(counts.sum())
    return {int(label): total / (2.0 * float(count)) for label, count in counts.items()}
