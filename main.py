from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
numpy.set_printoptions(threshold=numpy.nan)
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter


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
flags.DEFINE_float("mask_rate", 0.01, "fraction of depth data that can be seen\
        {0.01}")

"""     Network Architecture        """
flags.DEFINE_string("kernel_sizes", '3-3-3', "sizes of square kernel {3-3-3}")
flags.DEFINE_float("target_frac", 0.01, "fraction of points need to correct\
        {0.01}")
flags.DEFINE_string("gaussian_sizes", '11-11', "size of square gaussian kernel\
        {11-11}")
flags.DEFINE_integer("conv_size", 7, "size of square convolutional kernel {7}")

"""     Learning Parameters         """
flags.DEFINE_float("learning_rate", 1e-3, "learing rate {1e-3}")
flags.DEFINE_integer("max_epoch", 100, "max number of epochs to run {100}")
flags.DEFINE_integer("batch_size", 5, "#minibatch each batch {5}")
flags.DEFINE_float("lambda_1", 1.0, "weight of \|\hat{D}-D\|_2^2 {1.0}")
flags.DEFINE_float("lambda_2", 1.0, "weight of \|(\hat{D}-D)_{M^*}\|_2^2 {1.0}")
flags.DEFINE_float("lambda_3", 1.0, "weight of \
        \|(\hat{D}-D)_{M_{targets}}\|_2^2 {1.0}")

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

"""Return a Tensor G of shape [N, M]
    G(x,y) = 1.0/(2*pi*sigma^2) exp( -(x^2+y^2)/(2*sigma^2) )
"""
def gaussian_kernel(shape, dtype=tf.float32, partition_info=None, sigma=1.0):
    [N,M,x,y] = shape
    cx = (N-1)/2.0
    cy = (M-1)/2.0
    g = [] 
    sum_g = 0.0
    for x in range(N):
        g_x = []
        d_x = float(x)-cx
        for y in range(M):
            d_y = float(y)-cy
            entry = 1.0/(2*numpy.pi*sigma*sigma) * numpy.exp(
                    -(d_x*d_x+d_y*d_y)/(2*sigma*sigma))
            #print('dx=%f, dy=%f, entry=%f' % (d_x, d_y, entry))
            sum_g += entry
            g_x.append(entry)
        g.append(g_x)
    g/=sum_g
    #for x in range(N):
    #    for y in range(M):
    #        print(g[x,y], end=" ")
    #    print()
    return tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(g,
        dtype=tf.float32), axis=-1), axis=-1)

def estimate_from_sparse(Dstar, Mstar):
    conv_kernel_size = [FLAGS.conv_size, FLAGS.conv_size]
    gaussian_sizes = [[int(token), int(token)] for token in
            FLAGS.gaussian_sizes.split('-')]
    Dstar_ext = tf.expand_dims(Dstar, -1)
    Mstar_ext = tf.expand_dims(Mstar, -1)

    density = tf.layers.conv2d(
            inputs=Mstar_ext,
            filters=1,
            kernel_size=conv_kernel_size,
            strides=(1,1),
            padding='same',
            kernel_initializer=tf.constant_initializer(1.0),
            trainable=False,
            )
    conv = tf.layers.conv2d(
            inputs=Dstar_ext,
            filters=1,
            kernel_size=conv_kernel_size,
            strides=(1,1),
            padding='same',
            kernel_initializer=tf.constant_initializer(1.0),
            trainable=False,
            )

    conv = tf.div(conv, density+tf.constant(1e-20))
   
    gaussian = conv
    for gaussian_size in gaussian_sizes:
        gaussian = tf.layers.conv2d(
                inputs=gaussian,
                filters=1,
                kernel_size=gaussian_size,
                strides=(1,1),
                padding='same',
                kernel_initializer=gaussian_kernel,
                trainable=False,
                )

    return tf.squeeze(gaussian, axis=-1)

def random_mask(mask_rate):
    mask = tf.random_uniform(shape=[FLAGS.batch_size, FLAGS.height,
        FLAGS.width], minval=0, maxval=1, dtype=tf.float32)
    mask = tf.round(tf.sigmoid(mask - tf.constant(1.0-mask_rate)))
    return mask

def build_dataflow(data, indices):

    batch_data = tf.gather(data, indices)
    channels = tf.unstack(batch_data, axis=3)
    true_depth = channels[3]
    rgb = channels[:3]
    
    Mstar = random_mask(FLAGS.mask_rate)
    Dstar = tf.multiply(true_depth, Mstar)
    Dprime = estimate_from_sparse(Dstar, Mstar)
    Mstar_ext = tf.expand_dims(Mstar, axis=-1)
    Dstar_ext = tf.expand_dims(Dstar, axis=-1)
    Dprime_ext = tf.expand_dims(Dprime, axis=-1)

    Mtarget = random_mask(FLAGS.target_frac)
    Mtarget_ext = tf.expand_dims(Mtarget, axis=-1)
    Ddot = tf.multiply(true_depth, Mtarget)
    
    mix = Dprime + Ddot -tf.multiply(Dprime, Mtarget)

    inputs = tf.stack(rgb+[mix]+[Mstar], axis=3)

    assert inputs.shape.as_list() == [FLAGS.batch_size, FLAGS.height,
            FLAGS.width, FLAGS.channels+1]
    
    kernel_sizes = [[int(token), int(token)] for token in
            FLAGS.kernel_sizes.split('-')]

    last_layer = inputs

    for kernel_size in kernel_sizes:
        last_layer = hourglass(last_layer, 3, kernel_size)
        last_layer = tf.concat([last_layer, Dstar_ext, Mstar_ext], axis=-1)

    last_layer = tf.layers.conv2d(
            inputs=last_layer,
            filters=1,
            kernel_size=kernel_sizes[-1],
            strides=(1,1),
            padding='same',
            )

    true_depth_ext = tf.expand_dims(true_depth, axis=-1)

    summary_evals = {'generated':last_layer, 'real':true_depth_ext,
            'selected':Dstar_ext, 'blur':Dprime_ext}

    #compare_images = tf.concat([last_layer, ground_truth], axis=1)
    #rgb = tf.image.grayscale_to_rgb(compare_images)
    #rgb_summary = tf.summary.image('image', rgb, max_outputs=1)
    #rgb_summary = tf.summary.merge([rgb_summary])
    
    #print(last_layer.shape)
    loss_1 = FLAGS.lambda_1*tf.nn.l2_loss(last_layer-true_depth_ext)
    loss_2 = FLAGS.lambda_2*tf.nn.l2_loss(
            tf.multiply(last_layer-true_depth_ext, Mstar_ext))
    loss_3 = FLAGS.lambda_3*tf.nn.l2_loss(
            tf.multiply(last_layer-true_depth_ext, Mtarget_ext))

    loss = loss_1 + loss_2 + loss_3

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

"""Load variables and Return current epoch
Args:
    var_list: list of variables to restore
    load_dir: directory containing checkpoints, latest checkpoint is loaded

Returns:
    current_epoch: i.e. the suffix of the latest checkpoint file
"""
def load_vars(sess, var_list, load_dir):
    if len(var_list) == 0:
        return
    loader = tf.train.Saver(var_list)

    lc = tf.train.latest_checkpoint(load_dir+'/')
    if lc is not None:
        var_names = [v.name for v in var_list]
        print("restoring %s from %s" % (str(var_names), load_dir+'/'))
        loader.restore(sess, lc)
        return int(str(lc).split('step-')[-1])
    else:
        print('nothing exists in %s' % load_dir)
        return -1

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
    blurry_images = tf.placeholder(shape=[FLAGS.batch_size, FLAGS.height,
        FLAGS.width, 4], dtype=tf.float32)
    compare_images = tf.concat([real_images, gen_images, masked_images,
        blurry_images], axis=1)
    summary = tf.summary.image('image', compare_images, max_outputs=3)

    summary_writer = tf.summary.FileWriter(FLAGS.model_dir+'/tensorboard/')
    train_evals, summary_evals = build_dataflow(data, indices)
    saver = tf.train.Saver(tf.trainable_variables())


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        current_step = load_vars(sess, tf.trainable_variables(),
                FLAGS.model_dir)
        for i in range(current_step+1, 100000):
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
                mimgs = numpy.stack([generate_RGBA(vals['selected'][t, :, :, 0])
                    for t in range(FLAGS.batch_size)])
                bimgs = numpy.stack([generate_RGBA(vals['blur'][t, :, :, 0])
                    for t in range(FLAGS.batch_size)])
                summary_writer.add_summary(sess.run(summary,
                    feed_dict={real_images:rimgs, gen_images:gimgs,
                        masked_images:mimgs, blurry_images:bimgs}), i)
                saver.save(sess, FLAGS.model_dir+'/step', global_step=i)

if __name__ == "__main__":
    tf.app.run()
