import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.losses import losses_impl
import math

_ALPHA = 0.25
_SIGMA = 2.0
ALPHA = 0.25
SIGMA = 2.0

def _sigmoid(x):
	# y = 1 / (1 + exp(-x))
	negative_x = tf.multiply(x,tf.constant(-1.,dtype=float))
	denum = tf.math.add(
		tf.constant(1.,dtype=float),
		tf.math.exp(negative_x))
	return tf.div_no_nan(
		tf.constant(1.,dtype=float),
		denum)

def _conf_acc(y_true, y_pred):
	cy_labels,cx_labels,height_labels,width_labels,true_conf = tf.split(y_true,num_or_size_splits=5,axis=-1)
	cy_predictions,cx_predictions,height_predictions,width_predictions,pred_conf = tf.split(y_pred,num_or_size_splits=5,axis=-1)

	positives_mask = tf.where(
		math_ops.equal(true_conf, 1.0),
		array_ops.ones_like(true_conf), # positive match
		array_ops.zeros_like(true_conf), # negative match
	)

	#pred_conf_sigmoid = _sigmoid(pred_conf)
	bias_pred_conf = tf.math.add(pred_conf,tf.constant(-3.,dtype=float))
	pred_conf_sigmoid = tf.nn.sigmoid(bias_pred_conf)

	return tf.losses.compute_weighted_loss(
		K.abs(math_ops.subtract(positives_mask,pred_conf_sigmoid)),
		#positives_mask,
		#pred_conf_sigmoid,
		reduction=tf.losses.Reduction.MEAN)

def focal_loss(true_conf,pred_conf,sigma=SIGMA,reduction=tf.losses.Reduction.SUM,prior_bias=0.0,consider_unlabeled=True):
	''' Computes Focal Loss according to equation [4] in https://arxiv.org/abs/1708.02002
	FL(p_t) = -(1 - p_t)^sigma * log(p_t)
	'''
	
	# predictions are reduced before activation in order to represent a confidence prior
	bias_pred_conf = tf.math.add(pred_conf,tf.constant(prior_bias,dtype=float))
	
	#pred_conf_sigmoid = _sigmoid(bias_pred_conf) # between [0,1]
	pred_conf_sigmoid = tf.nn.sigmoid(bias_pred_conf) # between [0,1]
	
	def p_t_hat(y_true,y_pred):
		negative_mask = tf.where(
			math_ops.equal(y_true, -1.0),
			array_ops.ones_like(y_true), # positive match
			array_ops.zeros_like(y_true), # negative match
		)
		assigned_mask = tf.where(
			math_ops.not_equal(y_true, 0.0),
			array_ops.ones_like(y_true), # positive match
			array_ops.zeros_like(y_true), # negative match
		)
		p_t = tf.math.add(negative_mask,
			tf.multiply(y_pred,y_true))
		p_t_filter = tf.where(
			math_ops.equal(assigned_mask,1.0),
			p_t,
			array_ops.ones_like(p_t)
			)
		p_t_star = tf.where(
			tf.math.greater(p_t_filter,3e-700),
			p_t_filter,
			tf.math.add(
				array_ops.zeros_like(p_t_filter),
				tf.math.exp(-1600.0)))
		logit = tf.math.log(p_t_star)
		base = math_ops.subtract(
			tf.constant(1.0,dtype=float),
			p_t_filter)
			#p_t)
		return base,logit

	def p_t(y_true,y_pred):
		positives_mask = tf.where(
			math_ops.equal(y_true, 1.0),
			array_ops.ones_like(y_true), # positive match
			array_ops.zeros_like(y_true), # negative match
		)
		p_t = tf.where(
			math_ops.equal(positives_mask,1.0),
			y_pred,
			math_ops.subtract(
				array_ops.ones_like(y_pred),
				y_pred)
			)
		p_t_star = tf.where(
			tf.math.greater(p_t,3e-700),
			p_t,
			tf.math.add(array_ops.zeros_like(p_t),tf.math.exp(-1600.0)))
		logit = tf.math.log(p_t_star)
		base = math_ops.subtract(
			tf.constant(1.0,dtype=float),
			p_t)		
		return base,logit
	'''
	p_t = tf.multiply(
		math_ops.subtract(pred_conf_sigmoid,negative_mask),
		true_conf)
	'''
	if consider_unlabeled:
		base,logit = p_t(true_conf,pred_conf_sigmoid)
		weights = 1.0
	else:
		base,logit = p_t_hat(true_conf,pred_conf_sigmoid)
		weights = tf.where(
			math_ops.not_equal(true_conf, 0.0),
			array_ops.ones_like(true_conf), # positive match
			array_ops.zeros_like(true_conf), # negative match
		)

	exp = tf.constant(SIGMA,dtype=float)

	mod_factor = tf.pow(
		base,
		exp)

	losses = tf.multiply(
		tf.multiply(
			tf.constant(-1.,dtype=float),
			mod_factor),
		logit)

	focal_loss = tf.losses.compute_weighted_loss(
		losses,
		weights=weights,
		reduction=reduction)

	return focal_loss

def _focal_loss(y_true, y_pred):
	cy_labels,cx_labels,height_labels,width_labels,true_conf = tf.split(y_true,num_or_size_splits=5,axis=-1)
	cy_predictions,cx_predictions,height_predictions,width_predictions,pred_conf = tf.split(y_pred,num_or_size_splits=5,axis=-1)

	positives_mask = tf.where(
		math_ops.equal(true_conf, 1.0),
		array_ops.ones_like(true_conf), # positive match
		array_ops.zeros_like(true_conf), # negative match
	)

	classification_loss = focal_loss(
		true_conf,
		pred_conf,
		sigma=SIGMA,
		reduction=tf.losses.Reduction.SUM,
		prior_bias=0.0,
		consider_unlabeled=True,
		)
	
	weidhted_classification_loss = tf.multiply(tf.constant(ALPHA,dtype=float),classification_loss)

	normalizer = tf.div_no_nan(
		tf.constant(1.0,dtype=float),
		tf.reduce_sum(positives_mask),
		#tf.reduce_sum(assigned_mask),
		)

	loss = tf.multiply(normalizer,weidhted_classification_loss)
	return loss

def _focal_loss_ignore_unassigned(y_true, y_pred):
	cy_labels,cx_labels,height_labels,width_labels,true_conf = tf.split(y_true,num_or_size_splits=5,axis=-1)
	cy_predictions,cx_predictions,height_predictions,width_predictions,pred_conf = tf.split(y_pred,num_or_size_splits=5,axis=-1)

	positives_mask = tf.where(
		math_ops.equal(true_conf, 1.0),
		array_ops.ones_like(true_conf), # positive match
		array_ops.zeros_like(true_conf), # negative match
	)

	classification_loss = focal_loss(
		true_conf,
		pred_conf,
		sigma=SIGMA,
		reduction=tf.losses.Reduction.SUM,
		prior_bias=0.0,
		consider_unlabeled=False,
		)
	
	weidhted_classification_loss = tf.multiply(tf.constant(ALPHA,dtype=float),classification_loss)

	normalizer = tf.div_no_nan(
		tf.constant(1.0,dtype=float),
		tf.reduce_sum(positives_mask),
		#tf.reduce_sum(assigned_mask),
		)

	loss = tf.multiply(normalizer,weidhted_classification_loss)
	return loss

def _smooth_l1_loss(y_true, y_pred):
	cy_labels,cx_labels,height_labels,width_labels,true_conf = tf.split(y_true,num_or_size_splits=5,axis=-1)
	cy_predictions,cx_predictions,height_predictions,width_predictions,pred_conf = tf.split(y_pred,num_or_size_splits=5,axis=-1)

	positives_mask = tf.where(
		math_ops.equal(true_conf, 1.0),
		array_ops.ones_like(true_conf), # positive match
		array_ops.zeros_like(true_conf), # negative match
	)

	reg_y_loss = tf.losses.huber_loss(
		cy_labels,
		cy_predictions,
		weights=positives_mask,
		reduction=tf.losses.Reduction.SUM)
	
	reg_x_loss = tf.losses.huber_loss(
		cx_labels,
		cx_predictions,
		weights=positives_mask,
		reduction=tf.losses.Reduction.SUM)

	reg_h_loss = tf.losses.huber_loss(
		height_labels,
		height_predictions,
		weights=positives_mask,
		reduction=tf.losses.Reduction.SUM)

	reg_w_loss = tf.losses.huber_loss(
		width_labels,
		width_predictions,
		weights=positives_mask,
		reduction=tf.losses.Reduction.SUM)

	localization_loss = tf.math.add(tf.math.add(tf.math.add(reg_y_loss,reg_x_loss),reg_h_loss),reg_w_loss)

	#normalizer = tf.divide(
	normalizer = tf.div_no_nan(
		tf.constant(1.0,dtype=float),
		tf.reduce_sum(positives_mask),
		)
	
	loss = tf.multiply(normalizer,localization_loss)
	return loss

def combined_loss(y_true, y_pred):
	combined_loss = tf.math.add(
		_smooth_l1_loss(y_true, y_pred),
		_focal_loss(y_true, y_pred))

	return combined_loss

def combined_loss_ignore_unassigned(y_true, y_pred):
	combined_loss = tf.math.add(
		_smooth_l1_loss(y_true, y_pred),
		_focal_loss_ignore_unassigned(y_true, y_pred))

	return combined_loss

def _trivial_loss(y_true, y_pred):
	cY,cX,width,height,conf = tf.split(y_pred,num_or_size_splits=5,axis=-1)
	t_cY,t_cX,t_width,t_height,t_conf = tf.split(y_true,num_or_size_splits=5,axis=-1)
	
	mask = tf.constant([0.,0.,0.,0.,1.],dtype=float)
	
	positives_mask = tf.where(
		math_ops.equal(t_conf, 1.0),
		t_conf, # positive match
		array_ops.zeros_like(t_conf), # negative match
	)

	negative_mask = tf.where(
		math_ops.equal(t_conf, -1.0),
		array_ops.ones_like(t_conf), # positive match
		array_ops.zeros_like(t_conf), # negative match
	)

	loss = tf.losses.compute_weighted_loss(
		positives_mask,
		reduction=tf.losses.Reduction.SUM)

	loss = tf.reduce_mean(positives_mask)

	loss0 = tf.losses.compute_weighted_loss(
		K.abs(math_ops.subtract(y_pred, y_true)),
		reduction=tf.losses.Reduction.MEAN)

	loss1 = tf.losses.compute_weighted_loss(
		K.abs(math_ops.subtract(conf, t_conf)),
		reduction=tf.losses.Reduction.NONE)

	loss2 = tf.losses.compute_weighted_loss(
		K.abs(math_ops.subtract(tf.multiply(y_pred,mask), tf.multiply(y_true,mask))),
		reduction=tf.losses.Reduction.NONE,
		)
	
	# loss computed over total label vector (shape (5,)) with mask> to get correct mean
	fraction = tf.constant(5.,dtype=float)
	loss3 = tf.losses.compute_weighted_loss(
		tf.multiply(fraction,K.abs(math_ops.subtract(tf.multiply(y_pred,mask), tf.multiply(y_true,mask)))),
		reduction=tf.losses.Reduction.MEAN,
		)
	return loss
	#return tf.constant(0.8,dtype=tf.float32)
