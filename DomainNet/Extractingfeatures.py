import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import savetxt
import tensorflow as tf
from keras import applications

train_sketch = [os.path.join("domain_net-sketch/domain_net-sketch/domain_net-sketch_full/train",img) for img in os.listdir("domain_net-sketch/domain_net-sketch/domain_net-sketch_full/train")]
test_sketch = [os.path.join("domain_net-sketch/domain_net-sketch/domain_net-sketch_full/test",img) for img in os.listdir("domain_net-sketch/domain_net-sketch/domain_net-sketch_full/test")]
train_clipart = [os.path.join("domain_net-clipart/domain_net-clipart/domain_net-clipart_full/train",img) for img in os.listdir("domain_net-clipart/domain_net-clipart/domain_net-clipart_full/train")]
test_clipart = [os.path.join("domain_net-clipart/domain_net-clipart/domain_net-clipart_full/test",img) for img in os.listdir("domain_net-clipart/domain_net-clipart/domain_net-clipart_full/test")]

train_sketch_y = [int(img.split("/")[-1].split("_")[1]) for img in train_sketch]
test_sketch_y = [int(img.split("/")[-1].split("_")[1]) for img in test_sketch]
train_clipart_y = [int(img.split("/")[-1].split("_")[1]) for img in train_clipart]
test_clipart_y = [int(img.split("/")[-1].split("_")[1]) for img in test_clipart]

# load the ResNet101 network
print("[INFO] loading network...")
 
model = tf.keras.applications.ResNet101(weights="imagenet",include_top=False,pooling="avg")
model.summary()


def create_features(dataset, pre_model):
 
    x_scratch = []
 
    # loop over the images
    for imagePath in dataset:
 
        # load the input image and image is resized to 224x224 pixels
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
 
        # preprocess the image by (1) expanding the dimensions and
        # (2) subtracting the mean RGB pixel intensity from the
        # ImageNet dataset
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
 
        # add the image to the batch
        x_scratch.append(image)
 
    x = np.vstack(x_scratch)
    features = pre_model.predict(x, batch_size=32)
    features_flatten = features.reshape((features.shape[0], 2048))
    return x, features, features_flatten



train_sketch_x, train_sketch_features, train_sketch_features_flatten = create_features(train_sketch, model)
test_sketch_x, test_sketch_features, test_sketch_features_flatten = create_features(test_sketch, model)
train_clipart_x, train_clipart_features, train_clipart_features_flatten = create_features(train_clipart, model)
test_clipart_x, test_clipart_features, test_clipart_features_flatten = create_features(test_clipart, model)



savetxt('train_sketch_features.csv', train_sketch_features_flatten, delimiter=',')
savetxt('test_sketch_features.csv', test_sketch_features_flatten, delimiter=',')
savetxt('train_sketch_features.csv', train_clipart_features_flatten, delimiter=',')
savetxt('test_clipart_features.csv', test_clipart_features_flatten, delimiter=',')



train_sketch_y_1=np.asarray(train_sketch_y)
test_sketch_y_1=np.asarray(test_sketch_y)
train_clipart_y_1=np.asarray(train_clipart_y)
test_clipart_y_1=np.asarray(test_clipart_y)


savetxt('train_sketch_labels.csv', train_sketch_y_1, delimiter=',')
savetxt('test_sketch_labels.csv', test_sketch_y_1, delimiter=',')
savetxt('train_clipart_labels.csv', train_clipart_y_1, delimiter=',')
savetxt('test_clipart_labels.csv', test_clipart_y_1, delimiter=',')




