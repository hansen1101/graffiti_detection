''' Runs the training process for an object detection model according to a
configuration pipeline.

# Execution examples (call from Graffiti_Detection/ directory):
python train_inceptionV3_detector.py \
	--path_to_config_file=config/default_pipeline.yaml

# Args:
	-c --path_to_config_file path to a configuration file in yaml format
'''

import os
import sys
import getopt
import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
import yaml
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
import math
from keras.models import Model
from keras.layers import UpSampling2D,Input,Conv2D,Reshape,LeakyReLU,ZeroPadding2D,BatchNormalization

from utils import console_utils, data_sequence, extractAnnotations, classification_auxiliary, losses

import train_inceptionV3_base
import train_inceptionV3_classifier

DEFAULT_FEATURE_MAP_FILTER_SIZE = 256
DEFAULT_SIMILARITY_FUNCTION = 'jaccard_similarity'

def _linear_feature_pyramid(base_tensor_list,feature_map_filter_sizes,min_kernel_size=3,default_kernel_size=3,**kwargs):
	'''
	Args:
		base_tensor_list: list of tensors to build a feature pyramid upon
		feature_map_filter_sizes: list specifiying number of filters in successive feature map layers
	'''
	j = 0
	stride=2
	tmp_tensor = base_tensor_list[-1] # build new feature layers on last layer in feature map
	#for filter_size in feature_map_filter_sizes:
	if min_kernel_size <= 0:
		min_kernel_size = default_kernel_size
	while True:
		i = len(base_tensor_list)
		w = tmp_tensor.shape[1].value
		if w <= min_kernel_size:
			break
		kernel_size = min(w,default_kernel_size)
		padding = 0

		while True:
			target = (w-kernel_size+padding)/stride + 1
			if target % 2 == 1:
				break
			else:
				padding += 1

		if isinstance(feature_map_filter_sizes,list):
			filter_size = feature_map_filter_sizes[j]
			#print('filter size list')
			j += 1
		else:
			filter_size = feature_map_filter_sizes
			#print('single filter size')


		#((top_pad, bottom_pad), (left_pad, right_pad))
		if padding > 0:
			first_pad = math.ceil(padding/2)
			second_pad = math.floor(padding/2)
			assert first_pad + second_pad == padding
			tmp_tensor = ZeroPadding2D(padding=((first_pad, second_pad), (first_pad, second_pad)),name='fm_pad_{}_0'.format(i))(tmp_tensor)

		tmp_tensor = Conv2D(
			max(filter_size//2,1),
			(1,1),
			padding='valid',
			activation=None,
			name='fm_conv2d_{}_0'.format(i)
			)(tmp_tensor)

		tmp_tensor = BatchNormalization(name='fm_batchNorm_{}_0'.format(i))(tmp_tensor)

		tmp_tensor = LeakyReLU(alpha=0.15,name='fm_activation_{}_0'.format(i))(tmp_tensor)

		tmp_tensor =Conv2D(
			max(filter_size,1),
			(kernel_size,kernel_size),
			strides=(stride,stride),
			padding='valid',
			activation=None,
			name='fm_conv2d_{}_1'.format(i),
			)(tmp_tensor)

		tmp_tensor = BatchNormalization(name='fm_batchNorm_{}_1'.format(i))(tmp_tensor)

		tmp_tensor = LeakyReLU(alpha=0.15,name='fm_activation_{}_1'.format(i))(tmp_tensor)

		base_tensor_list.append(tmp_tensor)

	return base_tensor_list

def _generate_fpn(linaer_feature_pyramid,fpn_filter_sizes=DEFAULT_FEATURE_MAP_FILTER_SIZE,**kwargs):
	reverse_list = []
	tmp_tensor = None
	i=0
	while len(linaer_feature_pyramid) > 0:
		feature_tensor = Conv2D(fpn_filter_sizes, 1, strides=1, padding='same',name='fpn_{}_0'.format(i))(linaer_feature_pyramid.pop())
		if tmp_tensor is not None:
			tmp_tensor = UpSampling2D(size=(2,2),name='fpn_{}_up'.format(i))(tmp_tensor)
			# zero padding to fit to add
			target_shape = feature_tensor.shape[1].value
			padding = target_shape - tmp_tensor.shape[1].value
			if padding > 0:
				first_pad = math.ceil(padding/2)
				second_pad = math.floor(padding/2)
				assert first_pad + second_pad == padding
				tmp_tensor = keras.layers.add([feature_tensor,ZeroPadding2D(padding=((first_pad, second_pad), (first_pad, second_pad)),name='fpn_{}_pad'.format(i))(tmp_tensor)],name='fpn_{}_add'.format(i))
				reverse_list.append(tmp_tensor)
			elif padding < 0:
				# case tmp is larger than feat map
				first_pad = math.ceil(padding/(-2))
				second_pad = math.floor(padding/(-2))
				assert first_pad + second_pad == padding * (-1)
				tmp_tensor = keras.layers.add([ZeroPadding2D(padding=((first_pad, second_pad), (first_pad, second_pad)),name='fpn_{}_pad'.format(i))(feature_tensor),tmp_tensor],name='fpn_{}_add'.format(i))
				reverse_list.append(ZeroPadding2D(padding=((second_pad, first_pad), (second_pad, first_pad)),name='fpn_{}_pad_add'.format(i))(tmp_tensor))
			else:
				tmp_tensor = keras.layers.add([tmp_tensor,feature_tensor],name='fpn_{}_add'.format(i))
				reverse_list.append(tmp_tensor)
		else:
			tmp_tensor = feature_tensor
			reverse_list.append(tmp_tensor)
		i+=1
	#reverse_list.reverse()
	return reverse_list[::-1]

def _attach_prediction_head(feature_tensor_list,aspect_ratios,init_object_confidence=0.1,default_kernel_size=3,parameter_sharing=False,spatial_drop_out_rate=0.0,**kwargs):

	anchor_encoding_length = 5
	num_anchors_per_location = data_sequence.Detection_Data_Sequence._anchors_per_grid_cell(aspect_ratios)

	def _object_confidence_bias_initializer(shape, dtype=None):
		''' Initializes the bias for all object score coordinates to an initial score.
		The other coordinates are initialized with bias=0
		Args:
			shape: holds the shape of the layer (e.g. (?,15,15,40)
		Returns:
			kvar: bias variable with shape values
		'''
		val = np.zeros(shape,dtype=dtype)
		for i in range(val.shape[0]):
			if (i+1)%anchor_encoding_length==0:
				val[i]=init_object_confidence
		kvar = K.variable(value=val, dtype=dtype, name='bias_init')
		return kvar

	predict_tensor_list = []
	prediction_layer = Conv2D(
		num_anchors_per_location*anchor_encoding_length,
		(default_kernel_size,default_kernel_size),
		padding='same',
		#bias_initializer=_object_confidence_bias_initializer,
		name='pred_sharedParam_0',
		activation=None,
		#activation='sigmoid',
		#kernel_initializer=keras.initializers.Ones()
		)
	batchnorm_layer = BatchNormalization(name='pred_sharedParam_1')

	for i,feature_tensor in enumerate(feature_tensor_list):
		if spatial_drop_out_rate > 0.0:
			feature_tensor = keras.layers.SpatialDropout2D(spatial_drop_out_rate)(feature_tensor)
		if parameter_sharing:
			predict_tensor = prediction_layer(feature_tensor)
			predict_tensor = batchnorm_layer(predict_tensor)
		else:
			kernel_size = min(feature_tensor.shape[1].value,default_kernel_size)
			predict_tensor = Conv2D(
				num_anchors_per_location*anchor_encoding_length,
				(kernel_size,kernel_size),
				padding='same',
				#bias_initializer=_object_confidence_bias_initializer,
				name='pred_{}_0'.format(i),
				activation=None,
				#activation='sigmoid',
				#kernel_initializer=keras.initializers.Ones()
				)(feature_tensor)
			predict_tensor = BatchNormalization(name='pred_{}_1'.format(i))(predict_tensor)
		rank_1 = predict_tensor.shape[1].value
		rank_2 = predict_tensor.shape[2].value
		predict_tensor = Reshape(
			target_shape=(rank_1,rank_2,num_anchors_per_location,anchor_encoding_length),
			name='output{}'.format(i),
			)(predict_tensor)
		predict_tensor_list.append(predict_tensor)
	return predict_tensor_list

def _build_detection_model(base_model,base_tensor_list,aspect_ratios,feature_map_filter_sizes=DEFAULT_FEATURE_MAP_FILTER_SIZE,init_object_confidence=0.,feature_pyramid_network=False,**kwargs):
	''' Generates the detection model architecture according to the given
	configuration in the following order:
	1. freeze backbone cnn to bottleneck
	2. attach linear feature pyramid
	3. generate feature pyramid network, if configured
	4. attach detection head

	Args:
		base_model: keras backbone cnn
		base_tensor_list: list of tensors from the backbone cnn
			that should serve as base layers for the feature pyramid
		aspect_ratios: list of default anchor box aspect ratios
		feature_map_filter_sizes: depth of the linear feature maps
		init_object_confidence: currently not considered
		feature_pyramid_network: boolean indicates whether architecture should
			apply a feature pyramid network

	Retruns:
		keras detection model
	'''
	# Freeze model to last cnn base layer
	classification_auxiliary.freezeModelToBlock(
		base_model,
		[base_tensor_list[-1]],
		start_training_after_layers=True)
	if not feature_pyramid_network and kwargs.get('parameter_sharing',False):
		# linear feature pyramid with parameter sharing, append auxiliary layers
		# to base layers.
		print('[FPN:disabled]\ttransform base tensors from ',base_tensor_list,end='\t')
		transformed_base_tensor_list = []
		for i in range(len(base_tensor_list)):
			tensor = base_tensor_list[i]
			transfomed_tensor = Conv2D(
				max(feature_map_filter_sizes,1),
				(1,1),
				strides=(1,1),
				padding='same',
				activation=None,
				name='aux_fm_conv2d_{}_0'.format(i),
				)(tensor)
			transfomed_tensor = BatchNormalization(name='aux_fm_batchNorm_{}_0'.format(i))(transfomed_tensor)
			transfomed_tensor = LeakyReLU(alpha=0.15,name='aux_fm_activation_{}_0'.format(i))(transfomed_tensor)
			transformed_base_tensor_list.append(transfomed_tensor)
		base_tensor_list = transformed_base_tensor_list
		print('to ',base_tensor_list)
	feature_pyramid_tensors = _linear_feature_pyramid(
		base_tensor_list,
		feature_map_filter_sizes,
		**kwargs)
	print('linear pyramid contains: ',feature_pyramid_tensors)
	if feature_pyramid_network:
		feature_pyramid_tensors = _generate_fpn(
			feature_pyramid_tensors,
			**kwargs)
		print('fpn contains: ',feature_pyramid_tensors)
	prediction_tensors = _attach_prediction_head(
		feature_pyramid_tensors,
		aspect_ratios,
		init_object_confidence,
		**kwargs)
	return Model(inputs=base_model.inputs,outputs=prediction_tensors)

def _init_feature_map(model,feature_map_layers=[],**kwargs):
	''' Generates list of feature map tensors.
	'''
	feature_map_tensors = []
	for layer_name in feature_map_layers:
		layer = model.get_layer(layer_name)
		feature_map_tensors.append(layer.output)
	if not feature_map_tensors:
		print('[fallback to bottleneck layer] feature map layer field either '
			'unspecified in configuration file or model does not contain any '
			'layers with the specified names.')
		bottleneck_layer = kwargs.get('base_output_layer','')
		if bottleneck_layer:
			layer = model.get_layer(bottleneck_layer)
			feature_map_tensors.append(layer.output)
		else:
			print('[fallback to output tensor] bottleneck layer is also not '
				'contained in the model.')
			feature_map_tensors = model.outputs
	return feature_map_tensors

# @deprecated
def getModelDir(aspect_ratios,default_kernel_size,min_kernel_size,max_scale,min_scale,**kwargs):
	'''Generates the name of the model subdir from the detection config fields
	aspect_ratios, default_kernel_size, min_kernel_size, max_scale, min_scale

	Returns:
		list [1] of subdir name
	'''
	subdir_name = 'ar='
	for ar in aspect_ratios:
		subdir_name += '{:.1f}_'.format(ar)
	subdir_name = subdir_name[:-1]
	subdir_name += '__scale={:.2f}_{:.2f}'.format(min_scale,max_scale)
	subdir_name += '__kernel={}_{}'.format(default_kernel_size,min_kernel_size)
	return [subdir_name]

def getModelDirFromPipeline(subdir_name,spatial_drop_out_rate=None,**kwargs):
	''' Generates the detection subdir name '''
	if spatial_drop_out_rate is not None:
		return ['{}_drop_{:.1f}'.format(subdir_name,spatial_drop_out_rate)]
	else:
		return [subdir_name]

def init_weights(model,model_dir,weight_checkpoint_file='',**kwargs):
	''' Initializes the models weights from a checkpoint.

	Returns:
		epoch of the checkpoint (default:0)
	'''
	path = os.path.join(model_dir,weight_checkpoint_file)
	model.load_weights(os.path.join(model_dir,weight_checkpoint_file))
	print('weights loaded: ',weight_checkpoint_file)
	return 0

def generateModel(**config):
	''' Generates a keras detection model from a given config.
	Writes the model_config.yaml and initial weight checkpoint files
	to the model dir.

	Returns:
		model
		model_dir
		epoch
	'''
	getModelSubDir = train_inceptionV3_base.getModelDir_decorate(getModelDirFromPipeline,config)
	model_dir = getModelSubDir(**config['detection_config'])
	epoch = 0
	try:
		# Load a model from model_config.yaml and init weights from
		# weight_checkpoint_file field in the detection config part
		model = train_inceptionV3_base.model_from_yaml(model_dir,custom_objects={})
		epoch = init_weights(model,model_dir,**config['detection_config'])
		print('[OK] model could be restored')
	except Exception as e:
		# Build a new detection model from a classification model.
		#print(e)
		print('[FAIL] loading failed, fallback to base model')
		try:
			# try fallback to a base model if provided
			base_model_dir = getModelSubDir(subdir_name=config['base_model_dir'])
			print('try to generate base model from\t{}'.format(base_model_dir))
			base_model = train_inceptionV3_base.model_from_yaml(base_model_dir,custom_objects={})
			classification_epoch = config.get('pretrain_epoch',0)
			init_weights(base_model,base_model_dir,weight_checkpoint_file=config['base_model_ckpt'])
		except Exception as e1:
			# fallback to classification pipeline
			print(e1)
			base_model,_,classification_epoch = train_inceptionV3_classifier.generateModel(**config)
		base_feature_tensors = _init_feature_map(base_model,**config)
		model = _build_detection_model(
			base_model,
			base_feature_tensors,
			**config['detection_config'])
		print('[OK] classification model successfully generated')
		# save model description file
		train_inceptionV3_base.write_model_yaml(model,model_dir)

		# save initial weights
		model.save_weights(os.path.join(model_dir,'weights_init_classification_{:02d}.hdf5'.format(classification_epoch)))
		print('[OK] model architecture and initial weights stored in model dir')

	return model,model_dir,epoch

def freezeModel(model,layer_name):
	freeze_after_layer = isinstance(layer_name, str)
	if freeze_after_layer:
		after = False
		for layer in model.layers:
			layer.trainable = after
			if layer.name == layer_name:
				after = True
	else:
		print('freezing for list not implemented yet...')

	'''
	for layer in model.layers:
		print('{}\t{}\t{}'.format(layer.name,layer.trainable,layer.output))
	'''

def main(**kwargs):
	config = train_inceptionV3_base.load_config(kwargs['config_file_path'])
	model,model_dir,epoch = generateModel(**config)
	print(model_dir)
	# write config pipeline to model dir
	with open(os.path.join(model_dir,'pipeline.yaml'),'w') as f:
		f.write(yaml.dump(config))
	print('[OK] configuration pipeline in model dir')

	model_input_shape = (model.input_shape[1],model.input_shape[2],model.input_shape[3])
	config_input_shape = config['input_shape']
	assert model_input_shape == config_input_shape

	grid_sizes = []
	for output_tensor in model.outputs:
		grid_sizes.append(output_tensor.shape[1].value)

	print(grid_sizes)

	'''
	anchors = generate_anchor_list(
		grid_sizes,
		**config['detection_config'])

	linear_anchor_array = np.array(anchors)
	'''
	if config['detection_config'].get('enabled',True):
		data = extractAnnotations.process(
			bb_encoding_0_1=True,
			**config['detection_config'],
			)
		train_sequencer = data_sequence.Detection_Data_Sequence(
			data['train'],
			image_size=config['input_shape'],
			grid_sizes=grid_sizes,
			ignore_boundary_crossings=True,
			preprocess_on_init=True,
			**config['detection_config'])
		test_sequencer = data_sequence.Detection_Data_Sequence(
			data['test'],
			image_size=config['input_shape'],
			grid_sizes=grid_sizes,
			ignore_boundary_crossings=True,
			preprocess_on_init=True,
			**config['detection_config'])


		'''
		print('generators ready...')
		for i in range(len(train_sequencer)):
			x_i,y_i = train_sequencer[i]
			print(x_i.shape)
			for y_i_j in y_i:
				print(y_i_j.shape)
		sys.exit(0)
			__,y2 = test_sequencer[i]
			for j in range(len(y1)):
				y1_map = y1[j]
				y2_map = y2[j]
				#assert np.array_equal(y1_map,y2_map)
			print('comparison successful {}'.format(i))
		print('assertion done...')
		'''

		'''
		# check sequencer output
		for i in range(len(test_sequencer)):
			x,y = test_sequencer[i]
			for batch in y:
				for item in batch:
					for row in item:
						for cell in row:
							for anchor in cell:
								if anchor[4] in [-1.,0.]:
									assert np.all(anchor[:4]==0.)
								elif anchor[4] in [1.]:
									pass
								else:
									print(anchor[:])
		'''
		items_per_batch = config['detection_config'].get('batch_size',1)
		print('batch size:\t{}'.format(items_per_batch))
		print('training steps per epoch:\t{} (~{} items)\nvalidation steps per epoch:\t{} (~{} items)'.format(
			len(train_sequencer),
			items_per_batch*len(train_sequencer),
			len(test_sequencer),
			items_per_batch*len(test_sequencer)))

		losses.SIGMA = config['detection_config'].get('sigma',losses._SIGMA)
		loss_functions = config['detection_config']['loss']
		alpha_values = config['detection_config']['alpha']
		for loss_function_name in loss_functions:
			loss = getattr(losses,loss_function_name)
			for alpha_value in alpha_values:
				losses.ALPHA = alpha_value
				print('next training run with\n\t{}\n\t{}'.format(loss,losses.ALPHA))
				training_log = None
				epoch_count = 0
				epoch = init_weights(model,model_dir,**config['detection_config'])
				for i,training_stage in enumerate(config['detection_config'].get('trainings',[])):
					stage_epochs = training_stage.get('epochs',30)
					if training_stage.get('enabled',True):

						fine_tune_layer_name = config.get('base_output_layer','')
						if not training_stage.get('freeze_to_bottleneck',False):
							fine_tune_layer_name = training_stage.get('fine_tuning_layer',
								config.get('base_output_layer',''))
						print('freeze to {}'.format(fine_tune_layer_name))
						freezeModel(
							model,
							fine_tune_layer_name
							)

						model.compile(
							optimizer=train_inceptionV3_base.generateOptimizer(**training_stage),
							loss=loss,
							#loss_weights=[1.,1.1,1.5,1.,.75],
							metrics=[
								losses._smooth_l1_loss,
								losses._focal_loss,
								losses._focal_loss_ignore_unassigned,
								losses.combined_loss,
								losses.combined_loss_ignore_unassigned,
								],
						)

						callbacks = train_inceptionV3_base.generateCallbackList(**training_stage,model_dir=model_dir,loss_function_name=loss_function_name,alpha_value=alpha_value)

						# prompt model summary
						'''
						for layer in model.layers:
							print('{}\t{}'.format(layer.name,layer.trainable))
						sys.exit(0)
						model.summary()
						'''

						initial_epoch = max(epoch,epoch_count)
						print('initial epoch {}'.format(initial_epoch))

						first_training = model.fit_generator(
							generator=train_sequencer,
							steps_per_epoch=len(train_sequencer),
							epochs=epoch_count+stage_epochs,
							verbose=2,
							callbacks=callbacks,
							validation_data=test_sequencer,
							validation_steps=len(test_sequencer),
							initial_epoch=initial_epoch,
							#shuffle=True,
							)

						if training_log is None:
							training_log = {}
							for k,v in first_training.history.items():
								training_log[k] = v
						else:
							for k,v in first_training.history.items():
								training_log[k] += v

						epoch_count = initial_epoch
						try:
							if isinstance(first_training.history['val_loss'],list):
								length = len(first_training.history['val_loss'])
								epoch_count += length
								val_loss = first_training.history['val_loss'][length-1]
								print('final val loss: ',val_loss)
								model.save_weights(os.path.join(model_dir,'{}_alpha={:.2f}__weights_final__epoch={:02d}__val_loss={:.2f}.hdf5'.format(loss_function_name,alpha_value,epoch_count,val_loss)))
						except:
							print('final save failed')
					else:
						epoch_count += stage_epochs
						print('skip trainig stage {} to epoch {}'.format(i,epoch_count))

				with open(os.path.join(model_dir,'history__{}__alpha={:.2f}.yaml'.format(loss_function_name,alpha_value)),'w') as f:
					f.write(yaml.dump(training_log))

				# final evaluation
				model.compile(
					optimizer='sgd',
					loss=loss,
					#loss_weights=[1.2,1.,.9,.8],
					metrics=[
						#losses._conf_acc,
						losses._smooth_l1_loss,
						losses._focal_loss,
						],
					)
				predictions = model.evaluate_generator(
					generator=train_sequencer,
					steps=len(train_sequencer),
					verbose=1,
					)
				print(predictions)

if __name__ == '__main__':

	# extract parameters from console input and map along PARAMS
	options,long_options = console_utils.generate_opt_args_from_lists(train_inceptionV3_base.PARAMS)
	opts, args = getopt.getopt(
		sys.argv[1:],
		options,
		long_options,
		)

	# generate dict for extracted opts and args
	kwargs = train_inceptionV3_base.generate_kwargs_dict(opts,args)

	main(**kwargs)
