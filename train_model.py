import tensorflow as tf
from data import create_dataset, get_dataset_from_csv, mix_datasets, shuffle_data
from utils import evaluate, save_model_and_config, load_config, build_model, get_sigma, model_transfer_distance
from args import Args
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def map_onehot_labels(labels, map_to):
    n = labels.shape[1]
    raw_labels = np.argmax(labels, axis=1)
    new_labels = np.copy(raw_labels)
    for m, i in zip(map_to, range(len(map_to))):
        new_labels[raw_labels == i] = m
    new_labels = tf.one_hot(new_labels, depth=n)
    return new_labels


def train_model(args):
    # Create training and validation datasets
    if args.mix_datasets:
        target_path = args.data_path
        all_paths = args.mix_data_path
        all_paths.insert(0, target_path)
        x_train, y_train, w_train, x_val, y_val, x_test, y_test, args.num_classes = mix_datasets(rootdirs=all_paths,
                                                                                        subsamples=args.subsamples,
                                                                                        ws=args.weights,
                                                                                        num_classes=args.num_classes,
                                                                                        test_split=args.test_split,
                                                                                        val_split=args.val_split)
        x_train = tf.concat(x_train, axis=0)
        y_train = tf.concat(y_train, axis=0)
        w_train = tf.concat(w_train, axis=0)
        x_train, y_train, w_train = shuffle_data(x_train, y_train, w_train)

        x_val = x_val[0]
        y_val = y_val[0]

        x_test = x_test[0]
        y_test = y_test[0]

    else:
        x_train, y_train, x_val, y_val, x_test, y_test, args.num_classes = get_dataset_from_csv(rootdir=args.data_path,
                                                                                      val_split=args.val_split,
                                                                                      test_split=args.test_split,
                                                                                      shuffle=args.shuffle_before_split,
                                                                                      train_subsample=args.train_subsample,
                                                                                      num_classes=args.num_classes)
        w_train = None

    if args.map_test_labels is not None:
        y_val = map_onehot_labels(y_val, args.map_test_labels)
        y_test = map_onehot_labels(y_test, args.map_test_labels)

    if args.map_train_labels is not None:
        y_train = map_onehot_labels(y_train, args.map_train_labels)

    train_dataset = create_dataset(x_train, y_train, w=w_train, batch_sz=args.train_batch, shuffle=args.shuffle)
    val_dataset = create_dataset(x_val, y_val, batch_sz=args.val_batch, shuffle=False)
    test_dataset = create_dataset(x_test, y_test, batch_sz=args.val_batch, shuffle=False)

    args.input_shape = x_train.shape
    print('Input shape: ', args.input_shape)
    # Create model and optimizer
    if args.load_layer:  # Load architecture from source
        source_config = load_config(args.source_checkpoint)
        args.hidden_units = source_config.hidden_units
        args.use_bias = source_config.use_bias
        args.num_classes = source_config.num_classes
        assert source_config.architecture == 'onehidden'
        assert args.architecture == 'onehidden'
        src_model = build_model(source_config)
        src_model.load_weights(args.source_checkpoint)
        src_model._set_inputs(inputs=x_train)
        sigma_T = get_sigma(x_train, source_config.use_bias)

    model = build_model(args)
    model._set_inputs(inputs=x_train)  # Only needed in TF2.0 due to a bug in saving custom models. Will be fixed in TF2.2.

    # Load weights from source if needed for a single layer
    if args.load_layer:
        model.load_weights(args.source_checkpoint)
        model.train_only(args.layer_to_train)

    if args.freeze_output_layer:
        model.train_only('hidden')

    if args.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # Create loss function and metrics
    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    loss_metric = tf.keras.metrics.Mean()
    acc_metric = tf.keras.metrics.CategoricalAccuracy()

    # Compile model
    model.compile(optimizer, loss=mse_loss_fn)
    model.summary()

    train_loss_arr = []
    train_acc_arr = []
    val_loss_arr = []
    val_acc_arr = []

    # Train the model
    for epoch in range(args.epochs):
        loss_metric.reset_states()
        acc_metric.reset_states()
        for step, train_batch in enumerate(train_dataset):
            x_batch_train, y_batch_train = train_batch[:2]
            with tf.GradientTape() as tape:
                preds = model(x_batch_train)
                if args.mix_datasets:
                    w_batch_train = train_batch[2]
                    loss = mse_loss_fn(preds, y_batch_train, sample_weight=w_batch_train)
                else:
                    loss = mse_loss_fn(preds, y_batch_train)
            grads = tape.gradient(loss, model.trainable_weights)

            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            loss_metric.update_state(loss)
            acc_metric.update_state(y_batch_train, preds)

        if epoch % args.val_freq == 0:
            val_loss, val_acc = evaluate(model, val_dataset)
            val_loss_arr.append(val_loss)
            val_acc_arr.append(val_acc)
            print('val loss = %s val accuracy = %s' % (val_loss, val_acc))

        train_acc = acc_metric.result().numpy()
        train_loss_arr.append(loss_metric.result().numpy())
        train_acc_arr.append(train_acc)
        print('epoch %s: train mean loss = %s train accuracy = %s' % (epoch, loss_metric.result().numpy(),
                                                                      acc_metric.result().numpy()))
        if args.stop_at_acc is not None:
            if args.stop_at_acc <= acc_metric.result():
                print('Goal accuracy reached. Stopping training. ')
                break

    # Evaluate model
    val_loss, val_acc = evaluate(model, val_dataset)
    print('Final val loss = %s val accuracy = %s' % (val_loss, val_acc))
    val_loss_arr.append(val_loss)
    val_acc_arr.append(val_acc)

    # Save model and results
    save_dict = {'train_loss': np.array(train_loss_arr),
                 'train_acc': np.array(train_acc_arr),
                 'val_loss': np.array(val_loss_arr),
                 'val_acc': np.array(val_acc_arr),
                 'distance': None}

    if args.load_layer:
        distance = model_transfer_distance(src_model, model, sigma_T, source_config.use_bias)
        print('Final model distance: ', distance)
        save_dict['distance'] = distance

    args.save_dict = save_dict
    print('Saving model...')
    save_model_and_config(args.checkpoint_path, model, args, save_dict)
    print('Done!')


if __name__ == '__main__':
    args = Args().parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    train_model(args)

