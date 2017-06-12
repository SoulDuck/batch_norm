
from cnn import convolution2d , batch_norm_layer , affine , max_pool , convolution2d_manual
import tensorflow as tf
#ef convolution2d(name,x,out_ch,k=3 , s=2 , padding='SAME'):
def stem(name , x):
    with tf.variable_scope(name) as scope:
        layer=convolution2d('stem_cnn_0',x,32,k=3,s=2 , padding='VALID')
        layer = convolution2d('stem_cnn_1',layer, 32, k = 3, s = 1, padding = 'VALID')
        layer = convolution2d('stem_cnn_2', layer, 64, k=3, s=1, padding='SAME')
        layer_1 = max_pool('stem_max_3', layer, k=3, s=2, padding='VALID')
        layer_2 = convolution2d('stem_cnn_3_1', layer, 96, k=3, s=2, padding='VALID')
        layer_join=tf.concat([layer_1,layer_2] , axis=3 , name='join')
        print 'layer_name :','join'
        print 'layer_shape :',layer_join.get_shape()
    return layer_join
def stem_1(name , x ):
    with tf.variable_scope(name) as scope:
        layer = convolution2d('stem_cnn_0', x, 64, k=1, s=1)
        layer = convolution2d('stem_cnn_1', layer, 96, k=3, s=1, padding='VALID')
        layer_ = convolution2d('stem_cnn__0', x, 64, k=1, s=1)
        layer_ = convolution2d_manual('stem_cnn__1', layer_, 64, k_h=7,k_w=1, s=1)
        layer_ = convolution2d_manual('stem_cnn__2', layer_, 64, k_h=1,k_w=7,s=1 )
        layer_ = convolution2d('stem_cnn__3', layer_, 96, k=3, s=1, padding='VALID')
        layer_join = tf.concat([layer, layer_], axis=3, name='join')
    return layer_join
def stem_2(name ,x ):
    with tf.variable_scope(name) as scope:
        layer= convolution2d('stem_cnn_0' , x, 192,k=3,s=2,padding='VALID')
        layer_=max_pool('stem_max__0' , x, k=3 , s=2 , padding = 'VALID')
        layer_join = tf.concat([layer , layer_] , axis = 3 ,name='join')
    return layer_join



def reductionA(name,x ):

    with tf.variable_scope(name) as scope:
        layer_ =max_pool('max_pool_0' ,x, k=3, s=2 ,padding='VALID')

        layer__ =convolution2d('cnn__0' ,x,192 , k=3 , s=2 , padding='VALID'  )

        layer___ = convolution2d('cnn___0',x,224, k=1, s=1, padding='SAME')
        layer___ = convolution2d('cnn___1',layer___,256, k=3, s=1, padding='SAME')
        layer___ = convolution2d('cnn___2',layer___,385, k=3, s=2, padding='VALID')

        layer_join=tf.concat([layer_ , layer__ , layer___ ], axis=3 , name='join')
    return layer_join

def reductionB(name , x):
    with tf.variable_scope(name) as scope:
        layer_ = max_pool('max_pool_0',x, k=3, s=2, padding='VALID')

        layer__ = convolution2d('cnn__0',x,192, k=1, s=1, padding='SAME')
        layer__ = convolution2d('cnn__1' ,layer__, 192,k=3 ,s=2 ,padding='VALID')

        layer___ = convolution2d('cnn___0',x,256, k=1, s=1, padding='SAME')
        layer___ = convolution2d_manual('cnn___1',layer___,256, k_h=1 , k_w=7, s=1, padding='SAME')
        layer___ = convolution2d_manual('cnn___2',layer___,320, k_h=7, k_w=1, s=1, padding='SAME')
        layer___ = convolution2d('cnn___3',layer___, 320,k=3, s=2, padding='VALID')

        layer_join=tf.concat([layer_ , layer__ , layer___] , axis=3 , name='join')

    return layer_join
def blockA(name , x):


def blockB(name , x)

def blockC(name , x)