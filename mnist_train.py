#-*- coding:utf-8 -*-
import tensorflow as tf
from cnn  import convolution2d , max_pool , algorithm , affine , batch_norm_0 , batch_norm_1 , batch_norm_2 , logits
import data
import utils
import os
import numpy as np
from inception_v4 import  stem  , stem_1 , stem_2 , reductionB , reductionA ,blockA , blockB ,blockC ,resnet_blockA , resnet_blockB , resnet_blockC
##########################setting############################

image_height, image_width, image_color_ch, n_classes, train_imgs, train_labs, test_imgs, test_labs = data.mnist_28x28()
x_ = tf.placeholder(dtype=tf.float32, shape=[None, image_height, image_width, image_color_ch], name='x_')
y_ = tf.placeholder(dtype=tf.int32, shape=[None, n_classes], name='y_')
phase_train=tf.placeholder(dtype=tf.bool , name='phase_train')
batch_size=60
##########################structure##########################


layer , conv1_summary_tensor  = convolution2d('conv1', x_, 64)
layer = batch_norm_0(layer, phase_train, 'bn0')
layer = max_pool('max_pool1' , layer )
layer , topconv_summary_tensor = convolution2d('top_conv', layer, 128)
layer = batch_norm_0(layer, phase_train, 'bn1')
layer , fc_summary_tensor = affine('fully_connect', layer, 1024 ,keep_prob=0.5 ,phase_train= phase_train)
layer = batch_norm_0(layer, phase_train, 'bn2')
y_conv=logits('end_layer' , layer , n_classes)
merged = tf.summary.merge_all()

#############################################################
#cam = get_class_map('gap', top_conv, 0, im_width=image_width)
pred_op, pred_cls, cost, train_op, correct_pred, accuracy = algorithm(y_conv, y_, 0.1)
writer=tf.summary.FileWriter(logdir='./logs')
writer.add_graph(graph = tf.get_default_graph())
saver = tf.train.Saver()
sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
if not os.path.isdir('./cnn_model'):
    os.makedirs('./cnn_model')
try:
    saver.restore(sess, './cnn_model/best_acc.ckpt')
    print 'model was restored!'
except tf.errors.NotFoundError:
    print 'there was no model'
########################training##############################
max_val = 0
max_iter=7000
check_point = 50
batch_size = 60
f=utils.make_log_txt()
train_acc=0;train_loss=0;


share=len(test_labs)/batch_size
remainder=len(test_labs)/batch_size

for step in range(max_iter):
    val_acc_mean, val_loss_mean, pred_all = [], [], []

    if step % check_point ==0 :
        # train 과 batch size 을 똑같이 하고 평가합니다
        batch_size=60
        print 'Validation Batch Size : {} '.format(batch_size)
        for i in range(share):  # 여기서 테스트 셋을 sess.run()할수 있게 쪼갭니다
            test_feedDict = {x_: test_imgs[i * batch_size:(i + 1) * batch_size],
                             y_: test_labs[i * batch_size:(i + 1) * batch_size], phase_train: False}
            # check summary shape , and value
            val_acc, val_loss, pred = sess.run([accuracy , cost , pred_op ], feed_dict=test_feedDict)
            val_acc_mean.append(val_acc)
            val_loss_mean.append(val_loss)
            pred_all.append(pred)
        val_acc_mean = np.mean(np.asarray(val_acc_mean))
        val_loss_mean = np.mean(np.asarray(val_loss_mean))
        summary=sess.run(merged , feed_dict= test_feedDict)
        writer.add_summary(summary, step)
        conv1_summary, topconv_summary, fc_summary = sess.run(
            [conv1_summary_tensor, topconv_summary_tensor, fc_summary_tensor], feed_dict=test_feedDict)
        print 'conv1 summary : ', conv1_summary
        print 'topconv summary : ', topconv_summary
        print 'FC summary : ', fc_summary
        summary=tf.Summary(value=[tf.Summary.Value(tag='Test batch : {} loss'.format(batch_size), simple_value=float(val_loss_mean)),
                          tf.Summary.Value(tag='Test batch : {} acc'.format(batch_size), simple_value=float(val_acc_mean)),
                          tf.Summary.Value(tag='Train batch : {} loss'.format(batch_size), simple_value=float(train_loss)),
                          tf.Summary.Value(tag='Train batch : {} acc'.format(batch_size), simple_value=float(train_acc))])
        writer.add_summary(summary, step)



        # validation batch size 을 1 로 합니다
        print 'Validation Batch Size : 1 '
        for i in range(len(test_labs)):  # 여기서 테스트 셋을 sess.run()할수 있게 쪼갭니다

            test_feedDict = {x_: test_imgs[i * batch_size:(i + 1) * batch_size],
                             y_: test_labs[i * batch_size:(i + 1) * batch_size], phase_train: False}
            # check summary shape , and value
            val_acc, val_loss, pred = sess.run([accuracy , cost , pred_op ], feed_dict=test_feedDict)
            val_acc_mean.append(val_acc)
            val_loss_mean.append(val_loss)
            pred_all.append(pred)
        val_acc_mean = np.mean(np.asarray(val_acc_mean))
        val_loss_mean = np.mean(np.asarray(val_loss_mean))
        summary=tf.Summary(value=[tf.Summary.Value(tag='Test batch_size 1 loss', simple_value=float(val_loss_mean)),
                          tf.Summary.Value(tag='Test batch_size 1  acc', simple_value=float(val_acc_mean)),
                          tf.Summary.Value(tag='Train batch_size 1  loss', simple_value=float(train_loss)),
                          tf.Summary.Value(tag='Train batch_size 1  acc', simple_value=float(train_acc))])
        writer.add_summary(summary, step)

        print 'Train accuracy and loss :',train_acc , train_loss
        print 'Val accuracy and loss :', val_acc_mean ,val_loss_mean

    utils.show_progress(step,max_iter)
    batch_xs, batch_ys = data.next_batch(train_imgs, train_labs, batch_size)
    train_acc, train_loss, _ = sess.run([accuracy, cost, train_op], feed_dict={x_: batch_xs, y_: batch_ys , phase_train:True})

