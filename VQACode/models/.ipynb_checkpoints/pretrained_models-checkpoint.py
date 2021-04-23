import tensorflow as tf


def pretrained_cnn(cnn='inception', include_top=False):
    # Load a pretrained InceptionV3 model
    if cnn == 'inception':
        nn = tf.keras.applications.InceptionV3(include_top=include_top, weights='imagenet')
    elif cnn == 'resnet':
        nn = tf.keras.applications.ResNet50(include_top=include_top, weights='imagenet')
    elif cnn == 'resnet_v2':
        nn = tf.keras.applications.ResNet50V2(include_top=include_top, weights='imagenet')
    elif cnn == 'vgg19':
        nn = tf.keras.applications.VGG19(include_top=include_top, weights='imagenet')
    elif cnn == 'dense_net':
        nn = tf.keras.applications.DenseNet121(include_top=include_top, weights='imagenet')
    elif cnn == 'efficient_net':
        nn = tf.keras.applications.EfficientNetB7(include_top=include_top, weights='imagenet')
    else:
        raise ValueError
    # Set all layer.trainable False
    for layer in nn.layers:
        layer.trainable = False

    return nn


def cnn_preprocess_input(x, cnn=''):
    # preprocess the inputs for differnt types of CNNs
    if cnn == 'inception':
        return tf.keras.applications.inception_v3.preprocess_input(x)
    elif cnn == 'resnet':
        return tf.keras.applications.resnet50.preprocess_input(x)
    elif cnn == 'resnet_v2':
        return tf.keras.applications.resnet_v2.preprocess_input(x)
    elif cnn == 'vgg19':
        return tf.keras.applications.vgg19.preprocess_input(x)
    elif cnn == 'dense_net':
        return tf.keras.applications.densenet.preprocess_input(x)
    elif cnn == 'efficient_net':
        return tf.keras.applications.efficientnet.preprocess_input(x)
    else:
        return x / 255

