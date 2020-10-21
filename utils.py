import tensorflow as tf
import pickle
from pathlib import Path
import numpy as np
from models import OneHidden


def accuracy(pred, true):
    assert pred.shape == true.shape
    pred_label = tf.math.argmax(pred, axis=1)
    true_label = tf.math.argmax(true, axis=1)
    n = pred.shape[0]
    eq = tf.where(pred_label == true_label, 1., 0.)
    n_match = tf.reduce_sum(eq)
    return (n_match/n).numpy()


def evaluate(model, val_dataset, loss_fn=tf.keras.losses.MeanSquaredError()):
    loss_metric = tf.keras.metrics.Mean()
    loss_metric.reset_states()
    acc_arr = []
    for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
        preds = model(x_batch_val)
        loss = loss_fn(preds, y_batch_val)
        loss_metric.update_state(loss)
        acc_arr.append(accuracy(preds, y_batch_val))
    return loss_metric.result().numpy(), np.mean(acc_arr)


def save_model_and_config(checkpoint_path, model, config, losses=None):
    save_path = Path(checkpoint_path)
    model.save_weights(checkpoint_path)
    save_config(str(save_path.parents[0]) + '/config_'+str(save_path.name), config)
    if losses is not None:
        np.savez(str(save_path.parents[0]) + '/results_'+str(save_path.name), losses)


def load_config(checkpoint_path):
    save_path = Path(checkpoint_path)
    config_path = str(save_path.parents[0]) + '/config_'+str(save_path.name)
    with open(config_path, 'rb') as file:
        config = pickle.load(file)
        return config


def save_config(path, config):
        with open(path, 'wb') as file:
            pickle.dump(config, file)


def build_model(config):
    if config.architecture == 'onehidden':
        use_relu = True
    elif config.architecture == 'linear':
        use_relu = False
    m = OneHidden(hidden_units=config.hidden_units,
                  num_classes=config.num_classes,
                  use_bias=config.use_bias,
                  use_relu=use_relu)
    return m


def transfer_distance(W_S, W_T, sigma):
    diff = (W_S - W_T)
    result = np.matmul(np.transpose(diff), sigma)
    result = np.matmul(result, diff)
    return np.sqrt(np.trace(result))


def get_sigma(x, use_bias):
    if use_bias:
        x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
    sigma = np.cov(x, rowvar=False)
    return sigma


def model_transfer_distance(src_model, target_model, sigma_T, use_bias):
    if use_bias:
        W_S, b_S = src_model.get_hidden_layer_as_np()  # d x k, k x 1
        b_S = np.reshape(b_S, (1, -1))
        W_S = np.concatenate((W_S, b_S), axis=0)
        W_T, b_T = target_model.get_hidden_layer_as_np()
        b_T = np.reshape(b_T, (1, -1))
        W_T = np.concatenate((W_T, b_T), axis=0)
    else:
        W_S = src_model.get_hidden_layer_as_np()  # d x k, k x 1
        W_T = target_model.get_hidden_layer_as_np()
    sim = transfer_distance(W_S, W_T, sigma_T)
    return sim