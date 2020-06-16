import argparse


# Arguments for create_dataset.py
class DatasetArgs(argparse.ArgumentParser):

    def __init__(self, **overrides):

        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.add_argument('--data-dir', type=str, required=True,
                          help='Path to the folder where the images will be downloaded.')
        self.add_argument('--images-per-class', type=int, default=100,
                          help='Number of images per class in each dataset.')
        self.add_argument('--download-images', action='store_true',
                          help='If set, ImageNet images will be downloaded again. This deletes existing images! '
                               'Else, existing images from --data-dir will be used.')
        self.add_argument('--class-list', type=str, nargs='+',
                          help=' List of WordNet IDs for each class to be downloaded.')
        self.add_argument('--from-file', type=str, default=None,
                          help='If given, datasets will be downloaded based on the config file located here.'
                               'This overwrites every other settings.')


# Arguments for train_model.py and run_model.py
class Args(argparse.ArgumentParser):

    def __init__(self, **overrides):

        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.add_argument('--checkpoint-path', type=str, required=True,
                          help='Path to the folder to save the model after training or load the model for inference')
        self.add_argument('--data-path', type=str, required=True,
                          help='Path to the dataset used for training or evaluating the model')
        self.add_argument('--num-classes', default=None, type=int, help='Number of classes.')

        # Model parameters
        self.add_argument('--architecture', choices=['onehidden', 'linear'], default='onehidden',
                          help='Model architecture, in case of "onehidden" the model is f(x) = ReLU(x*W+b_1)*V + b2'
                               'x: input, (num_examples x input_dim)'
                               'W: hidden layer, (input_dim x hidden_units)'
                               'b1: hidden bias, (hidden_units,)'
                               'V: output layer, (hidden_units, num_classes)'
                               'b2: output bias, (num_classes,)'
                          'For "linear" there is no ReLU activation after the hidden layer.')
        self.add_argument('--hidden-units', default=32, type=int, help='Number of hidden units')
        self.add_argument('--use-bias', action='store_true',
                          help='If set, trainable bias term is added to the model in both input and output layer')

        # Transfer learning related
        self.add_argument('--freeze-output-layer', action='store_true',
                          help='If set, only the hidden layer will be trained')
        self.add_argument('--load-layer', action='store_true',
                          help='If set, layer weights for the layer set by --layer-to-load will be loaded '
                               'from --source-checkpoint')
        self.add_argument('--layer-to-train', choices=['hidden', 'output', 'both'], default='both',
                          help='Load weights from pretrained source model and only train over this layer.'
                               ' Only valid for one-hidden layer model')
        self.add_argument('--source-checkpoint', type=str,
                          help='Path to the folder to load weights from if --load-layer is set')

        # Training
        self.add_argument('--epochs', default=100, type=int, help='Number of training epochs')
        self.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam',
                          help='Optimizer used in training')
        self.add_argument('--lr', default=0.001, type=float, help='Learning rate')
        self.add_argument('--momentum', default=0.9, type=float, help='Momentum for SGD')
        self.add_argument('--train-batch', default=64, type=int, help='Training batch size')
        self.add_argument('--val-batch', default=64, type=int, help='Validation batch size')
        self.add_argument('--test-batch', default=64, type=int, help='Test batch size')
        self.add_argument('--train-subsample', default=1.0, type=float,
                          help='Fraction of training data used for training')
        self.add_argument('--val-split', default=0.1, type=float,
                          help='Fraction of training data used for validation')
        self.add_argument('--test-split', default=0.1, type=float,
                          help='Fraction of all data used for testing')
        self.add_argument('--shuffle-before-split', action='store_true',
                          help='If set, shuffle dataset before splitting into train and val')
        self.add_argument('--shuffle', action='store_true',
                          help='If set, shuffle train data after each epoch')
        self.add_argument('--stop-at-acc', default=None, type=float, help='If set, training is terminated'
                                                                          ' if train accuracy reaches this value.')
        self.add_argument('--val-freq', default=10, type=int, help='Validation frequency in epochs. '
                                                                   'Controls how frequently the model is validated.')

        # Mixing datasets
        self.add_argument('--mix-datasets', action='store_true',
                          help='If set, training dataset will be a mixture of'
                               ' --data-path and datasets listed in --mix-data-path')
        self.add_argument('--mix-data-path', type=str, nargs='+',
                          help=' List of paths to datasets to mix with. '
                               'NOTE: validation will be performed on the dataset given in --data-path.')
        self.add_argument('--subsamples', type=float, nargs='+', default=[0.0, 1.0],
                          help=' List of subsample ratios for mixing datasets. '
                               'This fraction of the training data will be used in the mixed training set. '
                               'Must have the same number of elements as all datasets to mix.')
        self.add_argument('--weights', type=float, nargs='+', default=[1.0, 1.0],
                          help=' List of weights in MSE loss for different datasets. '
                               'Must have the same number of elements as all datasets to mix.')

        # Misc.
        self.add_argument('--map-test-labels', type=float, nargs='+', default=None,
                          help=' Map test and validation labels to the labels given here. ')
        self.add_argument('--map-train-labels', type=float, nargs='+', default=None,
                          help=' Map train labels to the labels given here. ')
        self.add_argument('--seed', default=42, type=int, help='Seed for random number generators')

        self.set_defaults(**overrides)

