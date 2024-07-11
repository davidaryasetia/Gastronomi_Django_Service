import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import (
        vgg16,
        resnet50,
        ResNet50V2,
        MobileNet,
        inception_v3,
        InceptionResNetV2,
        ResNet152V2,
        ResNet101V2,
        MobileNetV2
    )

def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, keras.backend.epsilon()))

def siamese_architecture():
    input = keras.layers.Input((224, 224, 3))
    x = tf.keras.layers.BatchNormalization()(input)
    x = keras.layers.Conv2D(32, (3, 3), activation="tanh")(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = keras.layers.Conv2D(32, (5, 5), activation="tanh")(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = keras.layers.Dropout(0.5)(x) 
    x = keras.layers.Conv2D(64, (5, 5), activation="tanh")(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = keras.layers.Dropout(0.5)(x) 
    x = keras.layers.Conv2D(64, (7, 7), activation="tanh")(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = keras.layers.Dropout(0.5)(x) 
    x = keras.layers.Conv2D(128, (7, 7), activation="tanh")(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = keras.layers.Dropout(0.5)(x)

    x = keras.layers.Flatten()(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(10, activation="tanh")(x) #10 num class
    embedding_network = keras.Model(input, x)


    input_1 = keras.layers.Input((224, 224, 3))
    input_2 = keras.layers.Input((224, 224, 3))

    # As mentioned above, Siamese Network share weights between
    # tower networks (sister networks). To allow this, we will use
    # same embedding network for both tower networks.
    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    # input_1 = keras.layers.Input((224, 224, 3))
    # input_2 = keras.layers.Input((224, 224, 3))
    # mobilenet_model = MobileNet(weights='imagenet')
    # tower_1 = mobilenet_model(input_1)
    # tower_2 = mobilenet_model(input_2)

    merge_layer = keras.layers.Lambda(euclidean_distance)([tower_1, tower_2])
    normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
    output_layer = keras.layers.Dense(1, activation="sigmoid")(normal_layer)
    siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)
    return siamese