import tensorflow as tf
from pandas import DataFrame, Series


def to_dataset(
        X: DataFrame, y: Series,
        *, shuffle=True, batch_size=32) -> tf.data.Dataset:
    '''
    Load pandas objs as tf dataset with shuffle and batching
    '''
    df = X.copy()
    # df.items(): generator of tuple (col_name, series)
    # tf.newaxis: wrap value into additional dimension, i.e. from list[int] to list[list[int]]
    # df = {key: value[:, tf.newaxis] for key, value in [('series_name', pd.Series([1,2,3]))]} -> {'series_name': array([[1],[2],[3]])}
    df_dict = {key: value[:, tf.newaxis] for key, value in df.items()}
    # inspect ds with `next(ds.as_numpy_iterator())`:
    # ({'feature_name': feature_row_value}, label)
    ds = tf.data.Dataset.from_tensor_slices((df_dict, y))
    # shuffle elements; set buffer size greater than or equal to the full size of the dataset for perfect shuffling
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    # Combine 32 records into single batch
    ds = ds.batch(batch_size)
    # Allow prefetching 32 batches
    ds = ds.prefetch(batch_size)
    return ds
