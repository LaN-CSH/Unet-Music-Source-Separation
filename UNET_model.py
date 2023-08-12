import tensorflow as tf


def UNET(x, train):
    '''
    x = Input Batch
    tf.nn.leaky_relu :: leakiness(alpha) = 0.2 is the default value
    Convolution Layers all have kernel size [5,5], strides [2,2], padding 'same'
    '''

    ###################
    ####Convolution####
    ###################

    # Conv 1
    conv1 = tf.layers.conv2d(inputs=x, filters=16, kernel_size=[5, 5],
                             strides=[2, 2], padding="same", activation=tf.nn.leaky_relu)
    batch_conv1 = tf.layers.batch_normalization(conv1)
    print(batch_conv1.shape)

    # Conv 2
    conv2 = tf.layers.conv2d(inputs=batch_conv1, filters=32, kernel_size=[5, 5],
                             strides=[2, 2], padding="same", activation=tf.nn.leaky_relu)
    batch_conv2 = tf.layers.batch_normalization(conv2)
    print(batch_conv2.shape)
    # Conv 3
    conv3 = tf.layers.conv2d(inputs=batch_conv2, filters=64, kernel_size=[5, 5],
                             strides=[2, 2], padding="same", activation=tf.nn.leaky_relu)
    batch_conv3 = tf.layers.batch_normalization(conv3)
    print(batch_conv3.shape)
    # Conv 4
    conv4 = tf.layers.conv2d(inputs=batch_conv3, filters=128, kernel_size=[5, 5],
                             strides=[2, 2], padding="same", activation=tf.nn.leaky_relu)
    batch_conv4 = tf.layers.batch_normalization(conv4)
    print(batch_conv4.shape)
    # Conv 5
    conv5 = tf.layers.conv2d(inputs=batch_conv4, filters=256, kernel_size=[5, 5],
                             strides=[2, 2], padding="same", activation=tf.nn.leaky_relu)
    batch_conv5 = tf.layers.batch_normalization(conv5)
    print(batch_conv5.shape)
    # Conv 6
    conv6 = tf.layers.conv2d(inputs=batch_conv5, filters=512, kernel_size=[5, 5],
                             strides=[2, 2], padding="same", activation=tf.nn.leaky_relu)
    batch_conv6 = tf.layers.batch_normalization(conv6)
    print(batch_conv6.shape)
    ###################
    ###DeConvolution###
    ###################
    '''
    First 3 Layers have Dropout = 0.5
    Activation = ReLU
    Final Layer's activation function = sigmoid
    '''

    # DeConv 1
    deconv1 = tf.layers.conv2d_transpose(inputs=batch_conv6, filters=256, kernel_size=[5, 5],
                                         strides=[2, 2], padding="same", activation=tf.nn.relu)
    batch_deconv1 = tf.layers.batch_normalization(deconv1)
    dropout_deconv1 = tf.layers.dropout(inputs=batch_deconv1, rate=0.5, training=train)
    print(dropout_deconv1.shape)
    # DeConv 2
    concat_deconv2 = tf.concat([dropout_deconv1, batch_conv5], 3)
    deconv2 = tf.layers.conv2d_transpose(inputs=concat_deconv2, filters=128, kernel_size=[5, 5],
                                         strides=[2, 2], padding="same", activation=tf.nn.relu)
    batch_deconv2 = tf.layers.batch_normalization(deconv2)
    dropout_deconv2 = tf.layers.dropout(inputs=batch_deconv2, rate=0.5, training=train)
    print(dropout_deconv2.shape)
    # DeConv 3
    concat_deconv3 = tf.concat([dropout_deconv2, batch_conv4], 3)
    deconv3 = tf.layers.conv2d_transpose(inputs=concat_deconv3, filters=64, kernel_size=[5, 5],
                                         strides=[2, 2], padding="same", activation=tf.nn.relu)
    batch_deconv3 = tf.layers.batch_normalization(deconv3)
    dropout_deconv3 = tf.layers.dropout(inputs=batch_deconv3, rate=0.5, training=train)
    print(dropout_deconv3.shape)
    # DeConv 4
    concat_deconv4 = tf.concat([dropout_deconv3, batch_conv3], 3)
    deconv4 = tf.layers.conv2d_transpose(inputs=concat_deconv4, filters=32, kernel_size=[5, 5],
                                         strides=[2, 2], padding="same", activation=tf.nn.relu)
    batch_deconv4 = tf.layers.batch_normalization(deconv4)
    print(batch_deconv4.shape)
    # DeConv 5
    concat_deconv5 = tf.concat([batch_deconv4, batch_conv2], 3)
    deconv5 = tf.layers.conv2d_transpose(inputs=concat_deconv5, filters=16, kernel_size=[5, 5],
                                         strides=[2, 2], padding="same", activation=tf.nn.relu)
    batch_deconv5 = tf.layers.batch_normalization(deconv5)
    print(batch_deconv5.shape)
    # DeConv 6
    concat_deconv6 = tf.concat([batch_deconv5, batch_conv1], 3)
    deconv6 = tf.layers.conv2d_transpose(inputs=concat_deconv6, filters=4, kernel_size=[5, 5],
                                         strides=[2, 2], padding="same", activation=tf.nn.sigmoid)
    batch_deconv6 = tf.layers.batch_normalization(deconv6)
    print(batch_deconv6.shape)
    return batch_deconv6

