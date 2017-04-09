from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
numpy.set_printoptions(threshold=numpy.nan)
import tensorflow as tf
import matplotlib.pyplot as plt

flags = tf.flags
"""     System and File I/O         """
flags.DEFINE_string("data_path", 'data/merged_100.npy', "path to \
        train/test/valid data.")
flags.DEFINE_string("model_dir", '.', "where to output model and tensorboard")
flags.DEFINE_string("device", "1", "visible device")

"""     Data Specs                  """
flags.DEFINE_integer("height", 240, "height of images")
flags.DEFINE_integer("width", 320, "width of images")
flags.DEFINE_integer("channels", 4, "number of channels")
flags.DEFINE_float("mask_rate", 0.1, "fraction of depth data that can be seen")

"""     Network Architecture        """
flags.DEFINE_string("kernel_sizes", '5-7-9', "sizes of square kernel")

"""     Learning Parameters         """
flags.DEFINE_float("learning_rate", 1e-3, "learing rate {1e-3}")
flags.DEFINE_integer("max_epoch", 100, "max number of epochs to run {100}")
flags.DEFINE_integer("batch_size", 1, "#minibatch each batch {1}")

FLAGS = flags.FLAGS

import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.device

def hourglass(inputs, depth, kernel_size):
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
        #print('deconv%d.shape=%s'% (d, str(deconv_d.shape)))
        last_layer = deconv_d

    return last_layer 

def build_dataflow(data, indices):
    batch_data = tf.gather(data, indices)
    mask = tf.random_uniform(shape=[FLAGS.batch_size, FLAGS.height,
        FLAGS.width], minval=0, maxval=1, dtype=tf.float32)
    mask = tf.sigmoid(mask - tf.constant(1.0-FLAGS.mask_rate))
    mask = tf.round(mask)
    channels = tf.unstack(batch_data, axis=3)
    ground_truth = channels[3]
    masked = tf.multiply(ground_truth, mask)
    channels[3] = masked
    channels.append(mask)
    inputs = tf.stack(channels, axis=3)
    assert inputs.shape.as_list() == [FLAGS.batch_size, FLAGS.height,
            FLAGS.width, FLAGS.channels+1]
    
    kernel_sizes = [[int(token), int(token)] for token in
            FLAGS.kernel_sizes.split('-')]

    last_layer = inputs
    mask = tf.expand_dims(mask, axis=-1)
    masked = tf.expand_dims(masked, axis=-1)
    for kernel_size in kernel_sizes:
        last_layer = hourglass(last_layer, 3, kernel_size)
        last_layer = tf.concat([last_layer, masked, mask], axis=-1)

    last_layer = tf.layers.conv2d(
            inputs=last_layer,
            filters=1,
            kernel_size=kernel_sizes[-1],
            strides=(1,1),
            padding='same',
            )

    ground_truth = tf.expand_dims(ground_truth, axis=-1)

    summary_evals = {'generated':last_layer, 'real':ground_truth,
            'masked':masked}
    
    #compare_images = tf.concat([last_layer, ground_truth], axis=1)
    #rgb = tf.image.grayscale_to_rgb(compare_images)
    #rgb_summary = tf.summary.image('image', rgb, max_outputs=1)
    #rgb_summary = tf.summary.merge([rgb_summary])
    
    #print(last_layer.shape)
    loss = tf.nn.l2_loss(last_layer-ground_truth) + 1.0*tf.nn.l2_loss(
            tf.multiply(last_layer-ground_truth, mask))

    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    opt = optimizer.minimize(loss)

    #optimizer = tf.contrib.layers.optimize_loss(
    #        loss=loss,
    #        global_step=tf.contrib.framework.get_global_step(),
    #        learning_rate=FLAGS.learning_rate,
    #        optimizer="SGD"
    #        )
    train_evals = {'opt':opt, 'loss':loss}
    return train_evals, summary_evals

def generate_RGBA(img):
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=img.min(), vmax=img.max())
    return cmap(norm(img))

def main(_):
    print('batch_size=%d' % FLAGS.batch_size)
    data = numpy.load(FLAGS.data_path)
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    sample_size = data.shape.as_list()[0]
    print('total_sample_size=%d' % sample_size)
    shuffle = tf.random_shuffle(tf.range(0, limit=sample_size)) 
    shuffle_indices = shuffle[:FLAGS.batch_size]
    fixed_indices = tf.range(0, FLAGS.batch_size)
    indices = tf.placeholder(shape=[FLAGS.batch_size], dtype=tf.int32)

    real_images = tf.placeholder(shape=[FLAGS.batch_size, FLAGS.height,
        FLAGS.width, 4], dtype=tf.float32)
    gen_images = tf.placeholder(shape=[FLAGS.batch_size, FLAGS.height,
        FLAGS.width, 4], dtype=tf.float32)
    masked_images = tf.placeholder(shape=[FLAGS.batch_size, FLAGS.height,
        FLAGS.width, 4], dtype=tf.float32)
    compare_images = tf.concat([real_images, gen_images, masked_images], axis=1)
    summary = tf.summary.image('image', compare_images, max_outputs=3)

    summary_writer = tf.summary.FileWriter(FLAGS.model_dir+'/tensorboard/')
    train_evals, summary_evals = build_dataflow(data, indices)
    saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100000):
            vals = sess.run(train_evals,
                feed_dict={indices:sess.run(shuffle_indices)})
            print('iter=%d, loss=%f' % (i, vals['loss']))
            if i % 100 == 0:
                vals = sess.run(summary_evals,
                    feed_dict={indices:sess.run(fixed_indices)})
                gimgs = numpy.stack([generate_RGBA(vals['generated'][t, :, :,
                    0]) for t in range(FLAGS.batch_size)])
                rimgs = numpy.stack([generate_RGBA(vals['real'][t, :, :, 0]) for
                    t in range(FLAGS.batch_size)])
                mimgs = numpy.stack([generate_RGBA(vals['masked'][t, :, :, 0])
                    for t in range(FLAGS.batch_size)])
                summary_writer.add_summary(sess.run(summary,
                    feed_dict={real_images:rimgs, gen_images:gimgs,
                        masked_images:mimgs}), i)
                saver.save(sess, FLAGS.model_dir+'/step', global_step=i)

if __name__ == "__main__":
    tf.app.run()
