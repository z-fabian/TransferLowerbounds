import tensorflow as tf
import numpy as np


def csv_to_np(path):
    df = np.loadtxt(path)
    return df.astype(np.float32)


def center(x, axis=0):
    return x - tf.reduce_mean(x, axis=axis, keepdims=True)


def shuffle_data(x, y, w=None):
    n = x.shape[0]
    random_order = tf.random.shuffle(tf.range(n))
    x = tf.gather(x, random_order, axis=0)
    y = tf.gather(y, random_order, axis=0)
    if w is not None:
        w = tf.gather(w, random_order, axis=0)
        return x, y, w
    else:
        return x, y


def get_train_val_split(x, y, shuffle, val_split, train_subsample):
    assert 0 < val_split <= 1.0
    assert 0 < train_subsample <= 1.0
    if shuffle:
        x, y = shuffle_data(x, y)
    n = int(x.shape[0]*train_subsample)
    x = x[:n, ...]
    y = y[:n]
    n_val_data = int(x.shape[0] * val_split)
    x_val = x[:n_val_data, ...]
    y_val = y[:n_val_data, ...]
    x_train = x[n_val_data:, ...]
    y_train = y[n_val_data:, ...]
    return x_train, y_train, x_val, y_val


def subsample_classes(x, y, num_classes):
    indices = tf.where(tf.less(y, num_classes))[:, 0]
    indices = tf.cast(indices, tf.int32)
    x = tf.gather(x, indices, axis=0)
    y = tf.gather(y, indices, axis=0)
    return x, y


def get_dataset_from_csv(rootdir, val_split=0.1, test_split=0.1, shuffle=True, train_subsample=1.0, num_classes=None):
    x, y = csv_to_np(rootdir + '/features.csv'), csv_to_np(rootdir + '/labels.csv')
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.int32)

    x = center(x)  # Center distribution

    # Check and set number of classes
    total_classes = (tf.math.reduce_max(y)+1).numpy()  # Assuming integer labels starting from 0
    if num_classes is None:
        num_classes = total_classes
    assert num_classes <= total_classes

    n = x.shape[0]
    n_test = int(n * test_split)
    x_train, x_test = x[n_test:, ...], x[:n_test, ...]
    y_train, y_test = y[n_test:], y[:n_test]

    # Subsample classes
    x_train, y_train = subsample_classes(x_train, y_train, num_classes)
    x_test, y_test = subsample_classes(x_test, y_test, num_classes)

    # One-hot encode labels
    y_train = tf.one_hot(y_train, depth=num_classes)
    y_test = tf.one_hot(y_test, depth=num_classes)

    # Create train and validation datasets
    x_train, y_train, x_val, y_val = get_train_val_split(x_train,
                                                         y_train,
                                                         shuffle=shuffle,
                                                         val_split=val_split,
                                                         train_subsample=train_subsample)
    return x_train, y_train, x_val, y_val, x_test, y_test, num_classes


def mix_datasets(rootdirs, subsamples, ws, num_classes, test_split=0.1, val_split=0.1):
    assert len(rootdirs) == len(subsamples) == len(ws)

    x_test = []
    y_test = []
    x_val = []
    y_val = []
    x_train = []
    y_train = []
    train_weights = []

    for r, s, w in zip(rootdirs, subsamples, ws):
        x, y = csv_to_np(r + '/features.csv'), csv_to_np(r + '/labels.csv')

        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.int32)
        x = center(x)

        # Check and set number of classes
        if num_classes is None:
            num_classes = (tf.math.reduce_max(y) + 1).numpy()

        n_per_class_train = int(x.shape[0] * (1 - test_split) * (1 - val_split) * s // num_classes)
        n_per_class_val = int(x.shape[0] * (1 - test_split) * val_split // num_classes)
        n_per_class_test = int(x.shape[0] * test_split // num_classes)
        n_per_class = n_per_class_test + n_per_class_train + n_per_class_val

        temp_x_train = []
        temp_y_train = []
        temp_x_test = []
        temp_y_test = []
        temp_x_val = []
        temp_y_val = []

        for label in range(num_classes):
            x_filtered = tf.gather_nd(x, tf.where(y == label))
            y_filtered = tf.gather_nd(y, tf.where(y == label))
            y_filtered = tf.one_hot(y_filtered, depth=num_classes)
            temp_x_test.append(tf.identity(x_filtered[:n_per_class_test, ...]))
            temp_y_test.append(tf.identity(y_filtered[:n_per_class_test, ...]))
            temp_x_train.append(tf.identity(x_filtered[n_per_class_test:n_per_class_test + n_per_class_train, ...]))
            temp_y_train.append(tf.identity(y_filtered[n_per_class_test:n_per_class_test + n_per_class_train, ...]))
            temp_x_val.append(tf.identity(x_filtered[n_per_class_test + n_per_class_train:n_per_class, ...]))
            temp_y_val.append(tf.identity(y_filtered[n_per_class_test + n_per_class_train:n_per_class, ...]))

        train_weights.append(tf.ones(shape=(n_per_class_train * num_classes, 1)) * w)

        temp_x_train = tf.concat(temp_x_train, axis=0)
        temp_y_train = tf.concat(temp_y_train, axis=0)
        temp_x_test = tf.concat(temp_x_test, axis=0)
        temp_y_test = tf.concat(temp_y_test, axis=0)
        temp_x_val = tf.concat(temp_x_val, axis=0)
        temp_y_val = tf.concat(temp_y_val, axis=0)

        x_train.append(temp_x_train)
        y_train.append(temp_y_train)
        x_test.append(temp_x_test)
        y_test.append(temp_y_test)
        x_val.append(temp_x_val)
        y_val.append(temp_y_val)

    return x_train, y_train, train_weights, x_val, y_val, x_test, y_test, num_classes


def create_dataset(x, y, w=None, batch_sz=64, shuffle=False):
    if w is None:
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((x, y, w))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=x.shape[0])
    dataset = dataset.batch(batch_sz)
    return dataset

