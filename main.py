from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import tensorflow as tf

flags = tf.flags
"""     System and File I/O         """
flags.DEFINE_string("data_path", 'data/merged_100.npy', "path to train/test/valid data.")
flags.DEFINE_string("model_dir", None, "where to output model and tensorboard")
flags.DEFINE_string("device", None, "visible device")

"""     Data Specs                  """
flags.DEFINE_integer("height", 240, "height of images")
flags.DEFINE_integer("width", 320, "width of images")
flags.DEFINE_integer("channels", 4, "number of channels")

"""     Network Architecture        """
flags.DEFINE_string("kernel_size", '5,5', "size of kernel {5,5}")

"""     Learning Parameters         """
flags.DEFINE_float("learning_rate", 1e-3, "learing rate {1e-3}")
flags.DEFINE_integer("max_epoch", 100, "max number of epochs to run {100}")
flags.DEFINE_integer("batch_size", 10, "#minibatch each batch {1}")

FLAGS = flags.FLAGS

def build_dataflow(data):
    #indices = tf.placeholder(shape=[FLAGS.batch_size], dtype=tf.int32)
    sample_size = data.shape.as_list()[0]
    print('total_sample_size=%d' % sample_size)
    shuffle = tf.random_shuffle(tf.range(0, limit=sample_size))
    indices = shuffle[:FLAGS.batch_size]
    batch_data = tf.gather(data, indices)
    mask = tf.random_uniform(shape=[FLAGS.batch_size, FLAGS.height, FLAGS.width],
            minval=0, maxval=2, dtype=tf.int32)
    mask = tf.to_float(mask)
    channels = tf.unstack(batch_data, axis=3)
    grouth_truth = channels[3]
    channels[3] = tf.multiply(channels[3], mask)
    channels.append(mask)
    inputs = tf.stack(channels, axis=3)
    assert inputs.shape.as_list() == [FLAGS.batch_size, FLAGS.height, \
            FLAGS.width, FLAGS.channels+1]

    kernel_size = [int(token) for token in FLAGS.kernel_size.split(',')]

    conv1 = tf.layers.conv2d(
            inputs=inputs,
            filters=32,
            kernel_size=kernel_size,
            padding="same",
            activation=tf.nn.relu,
            )
    print('conv1.shape=%s'% str(conv1.shape))
    pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2,2],
            strides=2
            )

    print('pool1.shape=%s'% str(pool1.shape))
    
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=32,
            kernel_size=kernel_size,
            padding="same",
            activation=tf.nn.relu,
            )
    print('conv2.shape=%s'% str(conv2.shape))

    pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2,2],
            strides=2
            )

    print('pool2.shape=%s'% str(pool2.shape))

    conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=32,
            kernel_size=kernel_size,
            padding="same",
            activation=tf.nn.relu,
            )
    print('conv3.shape=%s'% str(conv3.shape))

    pool3 = tf.layers.max_pooling2d(
            inputs=conv3,
            pool_size=[2,2],
            strides=2
            )

    print('pool3.shape=%s'% str(pool3.shape))

    pool3_flat = tf.reshape(pool3, [FLAGS.batch_size, -1])

    print('pool3_flat.shape=%s'% str(pool3_flat.shape))
    
    dense1 = tf.layers.dense(inputs=pool3_flat, units=FLAGS.height*FLAGS.width,
            activation = tf.nn.relu)
    
    #dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)
    
    dense2 = tf.layers.dense(inputs=dense1, units=FLAGS.height*FLAGS.width,
            activation = tf.nn.relu)

    final = tf.reshape(dense2, [FLAGS.batch_size, FLAGS.height, FLAGS.width])

    loss = tf.nn.l2_loss(final-grouth_truth)

    optimizer = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer="SGD"
            )
    evals = {'opt':optimizer, 'loss':loss}
    return evals

def main(_):
    data = numpy.load(FLAGS.data_path)
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    evals = build_dataflow(data)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            vals = sess.run(evals)
            print('iter=%d, loss=%f' % (i, vals['loss']))

if __name__ == "__main__":
    tf.app.run()
