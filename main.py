from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import tensorflow as tf

flags = tf.flags
"""     System and File I/O         """
flags.DEFINE_string("data_path", 'data/merged_100.npy', "path to train/test/valid data.")
flags.DEFINE_string("model_dir", None, "where to output model and tensorboard")
flags.DEFINE_string("device", "1", "visible device")

"""     Data Specs                  """
flags.DEFINE_integer("height", 240, "height of images")
flags.DEFINE_integer("width", 320, "width of images")
flags.DEFINE_integer("channels", 4, "number of channels")

"""     Network Architecture        """
flags.DEFINE_string("kernel_size", '7,7', "size of kernel {7,7}")

"""     Learning Parameters         """
flags.DEFINE_float("learning_rate", 1e-3, "learing rate {1e-3}")
flags.DEFINE_integer("max_epoch", 100, "max number of epochs to run {100}")
flags.DEFINE_integer("batch_size", 10, "#minibatch each batch {10}")

FLAGS = flags.FLAGS

import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.device

def hourglass(inputs, depth):
    kernel_size = [int(token) for token in FLAGS.kernel_size.split(',')]

    last_layer = inputs

    for d in range(depth):
        conv_d = tf.layers.conv2d(
                inputs=last_layer,
                filters=32,
                strides=[2,2],
                kernel_size=kernel_size,
                padding="same",
                activation=tf.nn.relu,
                )
        print('conv%d.shape=%s'% (d, str(conv_d.shape)))
        last_layer = conv_d
    
    for d in range(depth):
        deconv_d = tf.layers.conv2d_transpose(
                inputs=last_layer,
                filters=32,
                strides=[2,2],
                kernel_size=kernel_size,
                padding="same",
                activation=tf.nn.relu,
                )
        print('deconv%d.shape=%s'% (d, str(deconv_d.shape)))
        last_layer = deconv_d

    return last_layer 

def build_dataflow(data):
    #indices = tf.placeholder(shape=[FLAGS.batch_size], dtype=tf.int32)
    sample_size = data.shape.as_list()[0]
    print('total_sample_size=%d' % sample_size)
    #shuffle = tf.range(0, limit=sample_size)
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

    last_layer = inputs
    for i in range(3):
        last_layer = hourglass(last_layer, 3)

    last_layer = tf.layers.conv2d(
            inputs=last_layer,
            filters=1,
            kernel_size=kernel_size,
            strides=(1,1),
            padding='same',
            )

    #print(last_layer.shape)
    loss = tf.nn.l2_loss(tf.squeeze(last_layer, axis=-1)-grouth_truth)

    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    opt = optimizer.minimize(loss)

    #optimizer = tf.contrib.layers.optimize_loss(
    #        loss=loss,
    #        global_step=tf.contrib.framework.get_global_step(),
    #        learning_rate=FLAGS.learning_rate,
    #        optimizer="SGD"
    #        )
    evals = {'opt':opt, 'loss':loss}
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
