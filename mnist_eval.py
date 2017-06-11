import tensorflow as tf
import data
sess=tf.Session()
saver=tf.train.import_meta_graph('./cnn_model/best_acc.ckpt.meta')
saver.restore(sess,'./cnn_model/best_acc.ckpt')
tf.get_default_graph()
softmax=tf.get_default_graph().get_tensor_by_name('softmax:0')
accuracy=tf.get_default_graph().get_tensor_by_name('accuracy:0')
#top_conv=tf.get_default_graph().get_tensor_by_name('top_conv/relu:0')
x_=tf.get_default_graph().get_tensor_by_name('x_:0')
y_=tf.get_default_graph().get_tensor_by_name('y_:0')
phase_train=tf.get_default_graph().get_tensor_by_name('phase_train:0')

if __name__=='__main__':
    image_height, image_width, image_color_ch, n_classes, train_imgs, train_labs, test_imgs, test_labs=data.mnist_28x28()
    pred=sess.run([accuracy], feed_dict={x_:test_imgs , y_:test_labs , phase_train : False})
    print pred
