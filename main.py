from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import tensorflow as tf

flags = tf.flags
"""     System and File I/O         """
flags.DEFINE_string("data_path", 'data/merged_100.npy', "path to train/test/valid data.")
flags.DEFINE_string("model_dir", '.', "where to output model and tensorboard")
flags.DEFINE_string("device", "1", "visible device")

"""     Data Specs                  """
flags.DEFINE_integer("height", 240, "height of images")
flags.DEFINE_integer("width", 320, "width of images")
flags.DEFINE_integer("channels", 4, "number of channels")
flags.DEFINE_float("mask_rate", 0.1, "fraction of depth data that can be seen")

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

def build_dataflow(data, indices):
    #indices = tf.placeholder(shape=[FLAGS.batch_size], dtype=tf.int32)
    sample_size = data.shape.as_list()[0]
    print('total_sample_size=%d' % sample_size)
    #shuffle = tf.range(0, limit=sample_size)
    batch_data = tf.gather(data, indices)
    mask = tf.random_uniform(shape=[FLAGS.batch_size, FLAGS.height, FLAGS.width],
            minval=0, maxval=1, dtype=tf.float32)
    mask = tf.sigmoid(mask - tf.constant(FLAGS.mask_rate))
    mask = tf.round(mask)
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

    compare_images = tf.concat([last_layer, tf.expand_dims(ground_truth)],
        axis=1)
    rgb = tf.image.grayscale_to_rgb(compare_images)
    rgb_summary = tf.summary.image('image', rgb)
    rgb_summary = tf.summary.merge([rgb_summary])
    
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
    train_evals = {'opt':opt, 'loss':loss}
    summary_evals = {'summary':rgb_summary}
    return train_evals, summary_evals

def main(_):
    print('batch_size=%d' % FLAGS.batch_size)
    data = numpy.load(FLAGS.data_path)
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    shuffle = tf.random_shuffle(tf.range(0, limit=sample_size)) 
    shuffle_indices = shuffle[:FLAGS.batch_size]
    fixed_indices = tf.range(0, FLAGS.batch_size)
    indices = tf.placeholder(shape=[FLAGS.batch_size], dtype=tf.int32)

    summary_writer = tf.summary.FileWriter(FLAGS.model_dir+'/tensorboard/')
    train_evals, summary_evals = build_dataflow(data, indices)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            vals = sess.run(train_evals,
                feed_dict={indices:sess.run(shuffle_indices)})
            print('iter=%d, loss=%f' % (i, vals['loss']))
            if i % 30 == 0:
                vals = sess.run(summary_evals,
                    feed_dict={indices:sess.run(fixed_indices)})
                summary_writer.add_summary(vals['summary'], i)

if __name__ == "__main__":
    tf.app.run()
