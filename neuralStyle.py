#coding=utf-8
import tensorflow as tf
import numpy as np
from scipy import misc
import vgg
import os

sess = tf.Session()

IMAGENET_MEAN_PIXEL = [ 123.68 , 116.779 , 103.939]


############################input params############################
INPUT_IMAGE_PATH="examples/input.jpg"
STYLE_IMAGE_PATH="examples/style1.jpg"
OUTPUT_IMAGE_PATH="generated"

###########################handle input shape#############################
content_image = misc.imread(INPUT_IMAGE_PATH)
#rescale style image according to content image
style_image = misc.imread(STYLE_IMAGE_PATH)
crows,ccols,cchannels = content_image.shape
srows,scols,schannels = style_image.shape
style_image = misc.imresize(style_image,1.0*ccols/scols)

#######################define content and style layers#####################
def get_style_content_feature(image_placeholder):
	vgg19, _ = vgg.net(image_placeholder)
	#style features
	style_feature_grams = []
	style_feature_layers = [vgg19['relu1_1'],vgg19['relu2_1'],vgg19['relu3_1'],vgg19['relu4_1'],vgg19['relu5_1']]
	for style_feature_layer in style_feature_layers:
		channels = style_feature_layer.get_shape().as_list()[-1]
		style_feature = tf.reshape(style_feature_layer,[-1,channels])
		style_feature_gram = tf.matmul(tf.transpose(style_feature),style_feature)/np.prod(style_feature.get_shape().as_list())
		style_feature_grams.append(style_feature_gram)
	#content features
	content_feature = vgg19['relu4_2']
	return style_feature_grams, content_feature
##########################utils function############################

def _tensor_size(tensor):
	from operator import mul
	return reduce(mul, (d.value for d in tensor.get_shape()), 1)

#################content and Style feature calculation##############

#calculate content features and style features
content_image = content_image.reshape((1,)+content_image.shape)
style_image = style_image.reshape((1,)+style_image.shape)

#get style feature from style_image
style_image_placeholder = tf.placeholder(tf.float32, style_image.shape)
style_feature_grams,_ = get_style_content_feature(style_image_placeholder)
style_features = [sess.run(x,feed_dict={style_image_placeholder:style_image-IMAGENET_MEAN_PIXEL}) for x in style_feature_grams]

#get content feature from content image
content_image_placeholder = tf.placeholder(tf.float32, content_image.shape)
_,content_feature_tensor = get_style_content_feature(content_image_placeholder)
content_feature = sess.run(content_feature_tensor,feed_dict={content_image_placeholder:content_image-IMAGENET_MEAN_PIXEL})



#######################Image Generation##################################
g = tf.Graph()
with g.as_default(), tf.Session() as sess:
	##initialize a image as output
	img = tf.random_normal(content_image.shape)
	image = tf.Variable(img, name="output_image",dtype=tf.float32)

	#create graph
	c_style_feature_grams, c_content_feature = get_style_content_feature(image)

	#calculate content loss: for reserve content
	c_loss = tf.nn.l2_loss(c_content_feature-content_feature)/content_feature.size
	
	#calculate style loss: for reserve style
	style_loss = 0
	for a,b in  zip(c_style_feature_grams, style_features):
		style_loss+=tf.nn.l2_loss(a-b)/b.size

	#total variation loss: for filter noises
	tdx = _tensor_size(image[:,1:,:,:])
	tdy = _tensor_size(image[:,:,1:,:])
	dx = image[:,1:,:,:]-image[:,:image.get_shape()[1]-1,:,:]
	dy = image[:,:,1:,:]-image[:,:,:image.get_shape()[2]-1,:]
	variation_loss = tf.nn.l2_loss(dx)/(tdx)+tf.nn.l2_loss(dy)/(tdy)

	#total loss
	total_loss = 10*c_loss+200*style_loss+200*variation_loss

	#train_op:only optimize image Variable
	train_op = tf.train.AdamOptimizer(learning_rate=100).minimize(total_loss,var_list=[image])

	sess.run(tf.initialize_all_variables())
	best_loss = np.inf
	for i in range(1000):
		# print image.eval()
		_, los = sess.run([train_op,total_loss])
		print "step:%s loss:%s:c_loss:%s style_loss:%s variation_loss:%s"%(str(i),los,c_loss.eval(),style_loss.eval(),variation_loss.eval())
		if los<best_loss:
			best_loss = los
			img = image.eval()+IMAGENET_MEAN_PIXEL
			#Attention!:Clip before save, otherwise will cause image blur
			img = np.clip(img[0], 0, 255).astype(np.uint8)
			misc.imsave(os.path.join(OUTPUT_IMAGE_PATH,"best_img.jpg"),img)
		if i%50==0:
			img = image.eval()+IMAGENET_MEAN_PIXEL
			#Attention!:Clip before save, otherwise will cause image blur
			img = np.clip(img[0], 0, 255).astype(np.uint8)
			misc.imsave(os.path.join(OUTPUT_IMAGE_PATH,"saved_img_%s.jpg"%str(i)),img)
