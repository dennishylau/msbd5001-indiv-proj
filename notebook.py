# %% import
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, losses, Model, metrics
from util import kfold, to_dataset
# from preprocessing import get_normalization_layer, get_category_encoding_layer
from preprocessing import encode_features
from datetime import datetime
# %%
df = pd.read_csv('./data/train.csv', index_col='id')
df.dropna(inplace=True)
y = df.pop('label')
X = df.copy()
# %%
X_train, X_test, y_train, y_test = kfold(X, y)
ds_train = to_dataset(X_train, y_train)
ds_test = to_dataset(X_test, y_test)

# %%

quant_feat = [
    'MO HLADR+ MFI (cells/ul)', 'Neu CD64+MFI (cells/ul)',
    'CD3+T (cells/ul)', 'CD8+T (cells/ul)', 'CD4+T (cells/ul)',
    'NK (cells/ul)', 'CD19+ (cells/ul)', 'CD45+ (cells/ul)', 'Age',
    'Mono CD64+MFI (cells/ul)'
]
cat_feat = ['Sex 0M1F']
all_inputs, encoded_features = encode_features(ds_train, quant_feat, cat_feat)

# %%
# merge the list of feature inputs `encoded_features`
# into one vector via concatenation with tf.keras.layers.concatenate
all_features = layers.concatenate(encoded_features)
x = layers.Dense(32, activation='relu')(all_features)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation='sigmoid')(x)
model = Model(all_inputs, output)

model.compile(
    optimizer='adam',
    loss=losses.BinaryCrossentropy(),
    metrics=[metrics.BinaryAccuracy()])

model.fit(ds_train, epochs=100, validation_data=ds_test)
loss, accuracy = model.evaluate(ds_test)

# predict

df_pred = pd.read_csv('./data/test.csv', index_col='id')
ds_pred = to_dataset(df_pred, None)
pred = model.predict(ds_pred)
pred_label = np.where(pred > 0.5, 1, 0)
df_pred['label'] = pred_label
df_pred['label'].to_csv(f'output/csv/{datetime.now()}_submission.csv')

# %% save and load model
model_name = f'{datetime.now()}.model'
model.save(f'./output/{model_name}')
# reloaded_model = models.load_model(f'./output/{model_name}')
