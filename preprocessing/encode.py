import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor
from .normalization import get_normalization_layer
from .categorical_encoding import get_category_encoding_layer


def encode_features(
    ds: tf.data.Dataset, quant_feat: list[str], cat_feat: list[str]
) -> tuple[list[KerasTensor], list[KerasTensor]]:

    all_inputs = []
    encoded_features = []

    for header in quant_feat:
        numeric_col = tf.keras.Input(shape=(1,), name=header)
        normalization_layer = get_normalization_layer(header, ds)
        encoded_numeric_col = normalization_layer(numeric_col)
        all_inputs.append(numeric_col)
        encoded_features.append(encoded_numeric_col)

    for header in cat_feat:
        cat_col = tf.keras.Input(shape=(1,), name=header, dtype='int64')
        encoding_layer = get_category_encoding_layer(
            name=header,
            dataset=ds,
            dtype='int64')
        encoded_col = encoding_layer(cat_col)
        all_inputs.append(cat_col)
        encoded_features.append(encoded_col)

    return all_inputs, encoded_features
