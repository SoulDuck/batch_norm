import tensorflow as tf
from cnn  import convolution2d , max_pool , algorithm , affine
import data
import batch
import utils
##########################setting############################

image_height, image_width, image_color_ch, n_classes, train_imgs, train_labs, test_imgs, test_labs = data.mnist_28x28()
x_ = tf.placeholder(dtype=tf.float32, shape=[None, image_height, image_width, image_color_ch], name='x_')
y_ = tf.placeholder(dtype=tf.int32, shape=[None, n_classes], name='y_')
phase_train=tf.placeholder(dtype=tf.bool , name='phase_train')
batch_size=60
##########################structure##########################
layer = convolution2d('conv1', x_, 64)
layer = max_pool(layer)
top_conv = convolution2d('top_conv', x_, 128)
layer = max_pool(top_conv)
layer=tf.contrib.layers.flatten(layer)
print layer.get_shape()
layer = affine('fully_connect', layer, 1024 ,keep_prob=0.5)
y_conv=affine('end_layer' , layer , n_classes , keep_prob=1.0)
#############################################################
#cam = get_class_map('gap', top_conv, 0, im_width=image_width)
pred, pred_cls, cost, train_op, correct_pred, accuracy = algorithm(y_conv, y_, 0.005)
saver = tf.train.Saver()
sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
try:
    saver.restore(sess, './cnn_model/best_acc.ckpt')
    print 'model was restored!'
except tf.errors.NotFoundError:
    print 'there was no model'
########################training##############################
max_val = 0
max_iter=10000
check_point = 300
for step in range(max_iter):
    utils.show_progress(step,max_iter)
    if step % check_point == 0:
        #inspect_cam(sess, cam, top_conv, test_imgs, test_labs, step, 50, x_, y_, y_conv)
        val_acc, val_loss = sess.run([accuracy, cost], feed_dict={x_: test_imgs[:100], y_: test_labs[:100]})
        print '\n',val_acc, val_loss
        if val_acc > max_val:

            #saver.save(sess, './cnn_model/best_acc.ckpt')
            print 'model was saved!'
    batch_xs, batch_ys = batch.next_batch(train_imgs, train_labs, batch_size)
    train_acc, train_loss, _ = sess.run([accuracy, cost, train_op], feed_dict={x_: batch_xs, y_: batch_ys})

