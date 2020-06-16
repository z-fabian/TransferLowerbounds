import tensorflow as tf
import os
import shutil
from args import DatasetArgs
import numpy as np
import subprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def download_imagenet_data(dataroot, class_list, images_per_class):
    subprocess.check_call(["python",
                           "external/ImageNet-Datasets-Downloader/downloader.py",
                           "-data_root", dataroot,
                           "-use_class_list", "True",
                           "-class_list"]+class_list+[
                           "-images_per_class", str(images_per_class)])


def extract_features(directory, sample_count, model, datagen, batch_size=1):
    generator = datagen.flow_from_directory(
        directory,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = model.predict(inputs_batch)
        feature_shape = features_batch.shape[1:]
        if i == 0:
            x = np.zeros(shape=(sample_count, feature_shape[0], feature_shape[1], feature_shape[2]))
            y = np.zeros(shape=(sample_count))
        x[i * batch_size: (i + 1) * batch_size] = features_batch
        y[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return x, y


def clean_datadir(data_dir):
    # Create download folder if doesn't exist and remove previous downloads
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    elif os.path.isdir(data_dir + '/imagenet_images'):
        shutil.rmtree(data_dir + '/imagenet_images')


def create_single_dataset(dataset_dir, class_list, images_per_class, model, datagen, download):
    num_classes = len(class_list)
    if download:
        download_imagenet_data(dataroot=dataset_dir,
                               class_list=class_list,
                               images_per_class=images_per_class)
    images_dir = os.path.join(dataset_dir, 'imagenet_images')
    if not os.path.isdir(images_dir):
        print('Images not found at ', dataset_dir, '. If you need to download images first, set --download-images.')
        return
    features, labels = extract_features(directory=images_dir,
                                        sample_count=images_per_class * num_classes,
                                        model=model,
                                        datagen=datagen)
    num = features.shape[0]
    features = np.reshape(features, newshape=(num, -1))
    labels = np.reshape(labels, newshape=(num, -1))
    np.savetxt(dataset_dir + '/features.csv', features, fmt='%.6f')
    np.savetxt(dataset_dir + '/labels.csv', labels, fmt='%d')


def create_datasets(args):
    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)

    # Load pretrained model for feature extraction
    model = tf.keras.applications.VGG16(weights='imagenet',
                                        include_top=False,
                                        input_shape=(224, 224, 3))
    datagen = ImageDataGenerator(rescale=1./255)

    if args.from_file is not None:
        with open(args.from_file) as f:
            for line in f.readlines():
                words = line.split()
                dataset_name, class_list, images_per_class = words[0], words[1:-1], int(words[-1])
                dataset_dir = os.path.join(args.data_dir, dataset_name)
                clean_datadir(dataset_dir)
                create_single_dataset(dataset_dir, class_list, images_per_class, model, datagen,
                                      download=args.download_images)
    else:
        assert hasattr(args, 'class_list')
        clean_datadir(args.data_dir)
        create_single_dataset(args.data_dir, args.class_list, args.images_per_class, model, datagen,
                              download=args.download_images)


if __name__ == '__main__':
    args = DatasetArgs().parse_args()
    create_datasets(args)







