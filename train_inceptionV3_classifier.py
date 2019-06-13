''' Runs the training process for an image classification model according to a
configuration pipeline.

# Execution examples (call from Graffiti_Detection/ directory):
python train_inceptionV3_classifier.py \
	--path_to_config_file=config/default_pipeline.yaml

# Args:
	-c --path_to_config_file path to a configuration file in yaml format
'''

import os
import keras
import sys
import re
import json
import yaml
import getopt
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, model_from_yaml
from keras.layers import Flatten, GlobalAveragePooling2D, Dropout, Dense, AveragePooling2D, MaxPooling2D, LeakyReLU, Activation, Conv2D, ZeroPadding2D, BatchNormalization
from keras.preprocessing import image
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, ProgbarLogger, TensorBoard
from keras.optimizers import SGD
from utils import console_utils
from utils.classification_auxiliary import tf_preprocessor, freezeModelToBlock
from keras.models import model_from_yaml
import train_inceptionV3_base

def generateCallbackList(training_config,model_dir):
	callbacks = []
	for className,kwargs in training_config.get('callbacks',{}).items():
		callback = getattr(keras.callbacks,className)
		if callback is TensorBoard:
			kwargs['log_dir']=os.path.join(model_dir,kwargs['log_dir'])
		elif callback is ModelCheckpoint:
			kwargs['filepath']=os.path.join(model_dir,kwargs['filepath'])
		callbacks.append(callback(**kwargs))
	return callbacks

def generateOptimizer(training_config):
	for className,kwargs in training_config.get('optimizer',{}).items():
		optimizer = getattr(keras.optimizers,className)
		return optimizer(**kwargs)

def single_class_binary_iterator(data_iterator,class_index=0.):
	for X_batch,Y_batch in data_iterator:
		yield X_batch,np.array([class_index for y in Y_batch])

def _generateHash(averagePooling2D=0,maxPooling2D=0,dense=[],**kwargs):
	dense_hash = '_'.join(['%d-%s'%(tup[0],tup[1]) for tup in dense])
	model_hash = 'avgPool2D=%d_maxPool2D=%d__%s'%(
		averagePooling2D,
		maxPooling2D,
		dense_hash,
		)
	return model_hash

def getModelDir(classification_config,**kwargs):
	# generate model directory
	head_layers = classification_config['head_layers']
	model_hash = _generateHash(**head_layers)

	dropout_folder_name = 'dropout__%s'%('_'.join(['%.2f' % rate for rate in head_layers['dropout']]))

	model_dir = os.path.join(train_inceptionV3_base.base_directory(**kwargs),model_hash,dropout_folder_name)
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)
	return model_dir

def _build_classification_model(base_model,base_output_layer,classification_config,**kwargs):
	''' Attaches the classification head to the model's base output layer.
 	The classification head is constructed according the classification
	configuration.

	Args:
		base_model keras model
		base_output_layer name of the output layer to which the classification
			head should be attached
		classification_config dictionary containing a 'head_layers' dictionary
			which has the following fields:
			- conv2d: list of tuples of keras.layers.Conv2D params
				(filters, kernel_size, strides, padding, activation, alpha)
			- averagePooling2D: boolean (default:0)
			- maxPooling2D: boolean (default:0)
			- dense: list of tuples of keras.layers.Dense params
				(units, activation, alpha)
			- dropout: list of dropout rates

	Returns:
		keras classification model
	'''
	x = base_model.output
	config = classification_config['head_layers']
	# change bottleneck features if base_output_layer flag is set
	if base_output_layer:
		print('Try to put classification head on layer \'{}\' ...'.format(base_output_layer))
		try:
			x = base_model.get_layer(base_output_layer).output
			print('   ... head successfully classification pipeline appended to layer \'{}\''.format(base_output_layer))
		except ValueError:
			base_output_layer = ''
			print('   ... output layer with name \'%s\' not found. Append head to base models default output tensor \'%s\' instead.'%(base_output_layer,head_input_tensor.name))

	# start building layers
	conv2d = config.get('conv2d',[]) # not implemented
	for i in range(len(conv2d)):
		params = conv2d[i]
		filters = params[0]
		kernels = params[1]
		strides = params[2]
		padding = params[3]
		mode = params[4]
		if len(params) > 5:
			activation_param = params[5]
		args = {}
		if len(params) > 6:
			args = params[6]
		if 'padding' in args:
			x = ZeroPadding2D(padding=args['padding'])(x)
		x = Conv2D(filters,kernels,strides=strides,padding=padding)(x)
		x = BatchNormalization()(x)
		if mode == 'relu':
			if 'name' in args:
				if isinstance(activation_param,dict):
					x = LeakyReLU(**activation_param,name=args['name'])(x)
				elif isinstance(activation_param,list) or isinstance(activation_param,tuple):
					x = LeakyReLU(*activation_param,name=args['name'])(x)
				else:
					x = LeakyReLU(alpha=0.15,name=args['name'])(x)

			else:
				if isinstance(activation_param,dict):
					x = LeakyReLU(**activation_param)(x)
				elif isinstance(activation_param,list) or isinstance(activation_param,tuple):
					x = LeakyReLU(*activation_param)(x)
				else:
					x = LeakyReLU(alpha=0.15)(x)
		else:
			if 'name' in args:
				x = Activation(activation=mode,name=args['name'])(x)
			else:
				x = Activation(activation=mode)(x)
	avg_pool = config.get('averagePooling2D',0)
	if avg_pool:
		x = AveragePooling2D()(x)

	max_pool = config.get('maxPooling2D',0)
	if max_pool:
		x = MaxPooling2D()(x)

	x = Flatten()(x)

	dense = config.get('dense',[])
	dropout = config.get('dropout',[])
	for i in range(len(dense)):
		params = dense[i]
		size = params[0]
		mode = params[1]
		if len(params) >= 3:
			activation_param = params[2]
		try:
			drop_rate = dropout[i]
		except:
			drop_rate = 0
		if mode == 'relu':
			x = Dense(size)(x)
			if isinstance(activation_param,dict):
				x = LeakyReLU(**activation_param)(x)
			elif isinstance(activation_param,list) or isinstance(activation_param,tuple):
				x = LeakyReLU(*activation_param)(x)
			else:
				x = LeakyReLU(alpha=0.15)(x)
		else:
			x = Dense(size)(x)
			x = Activation(activation=mode)(x)
		if drop_rate:
			x = Dropout(drop_rate)(x)
	return Model(inputs=base_model.input, outputs=x)

def generateInceptionV3BaseModel(input_shape,**kwargs):
	''' Loads a pre-trained InceptionV3 base model that
	accepts the given input shape. The classification head
	is detached from the network and pooling is disabled.

	Args:
		input_shape: tuple of (height,widht,depth)

	Returns:
		base_model keras model
	'''
	base_model = InceptionV3(
		weights='imagenet',
		include_top=False,
		input_shape=input_shape,
		pooling=None, # None: mixed10 output (None, 8, 8, 2048), 'avg' (None, 2048)
		)
	return base_model

def freezeModel(model,freeze_list=[]):
	# freeze base_model according to config
	freeze_after_layer = isinstance(freeze_list, str)
	if freeze_after_layer:
		after = False
		for layer in model.layers:
			layer.trainable = after
			if layer.name == freeze_list:
				after = True

	else:
		freezeModelToBlock(
			model,
			freeze_list,
			start_training_after_layers=freeze_after_layer,
			)

	'''
	for layer in model.layers:
		print('{}\t{}\t{}'.format(layer.name,layer.trainable,layer.output))
	'''

def generateModel(**kwargs):
	# load model config
	model_dir = getModelDir(**kwargs)
	epoch = 0
	print('classification model directory\t{}'.format(model_dir))
	try:
		with open(os.path.join(model_dir,train_inceptionV3_base.MODEL_YAML_FILE),'r') as f:
			yaml_string = f.read()

		# generate model
		model = model_from_yaml(yaml_string)
		print('[OK] provided configuration loaded')

		try:
			weight_ckpt_file_name = kwargs['classification_config'].get('weight_checkpoint_file','')
			weight_ckpt_path = os.path.join(model_dir,weight_ckpt_file_name)
			model.load_weights(weight_ckpt_path)
			print('[OK] provided checkpoint loaded')
		except:
			print('[FAIL] provided checkpoint could not be loaded, use fallback')
			epoch = init_weights(model,model_dir,**kwargs['classification_config'])
			print('[OK] fallback checkpoint loaded')
		print('[OK] classification model could be restored')
	except Exception as e:
		#print(e)
		print('[FAIL] model could not be restored, fallback to base model')
		# either model yaml not available or weiths could not be initialized
		base_model = generateInceptionV3BaseModel(**kwargs)

		model = _build_classification_model(
			base_model,
			**kwargs)
		print('[OK] classification model successfully generated')
		# save model description file
		with open(os.path.join(model_dir,train_inceptionV3_base.MODEL_YAML_FILE),'w') as f:
			f.write(model.to_yaml())

		# save initial weights
		model.save_weights(os.path.join(model_dir,'weights_init_imagenet.hdf5'))
		print('[OK] configuration and initial weights written to model dir')

	return model,model_dir,epoch

def init_weights(model,model_dir,weight_checkpoint_file='',**kwargs):

	weight_file_pattern = re.compile('^(weights|weights_final|weights_init_imagenet)(__epoch=(\d+))?(__val_loss=([\d]+\.[\d]+))?\.hdf5')
	files = os.listdir(model_dir)
	items = []
	latest = -1
	epoch = 0
	try:
		model.load_weights(os.path.join(model_dir,weight_checkpoint_file))
		m = re.match(weight_file_pattern,weight_checkpoint_file)
		if m:
			slug,_,epoch,_,val_loss = m.groups()
		print('loading provided weights from ckpt file {}'.format(weight_checkpoint_file))
	except:
		for i,file in enumerate(files):
			m = re.match(weight_file_pattern,file)
			if m:
				slug,_,epoch,_,val_loss = m.groups()
				try:
					epoch = int(epoch)
				except TypeError:
					epoch = 0
				try:
					val_loss=float(val_loss)
				except:
					val_loss=0.
				j = len(items)
				tup = (file,val_loss,epoch,)
				items.append(tup)
				if latest < 0:
					latest = j
				elif items[latest][2] < epoch:
					latest = j

				if slug == 'weights_final':
					latest = j
					break

		if 0<=latest<len(items):
			print('loading weights: %s'%items[latest][0])
			model.load_weights(os.path.join(model_dir,items[latest][0]))
			epoch = items[latest][2]
		else:
			# no weight file found > raise exception
			raise Exception('no weights file found')

	if epoch is None:
		epoch = 0
	return epoch

def compareModelWeights(model1,model2):
	for i,layer1 in enumerate(model1.layers):
		try:
			layer2 = model2.get_layer(index=i)
			if layer2.name != layer1.name:
				print('different layer names ',layer2.name,layer1.name)
			weights1 = layer1.get_weights()
			weights2 = layer2.get_weights()
			if len(weights1) <= 0 or len(weights2) <= 0:
				#print('no weights found for layer: ',layer1.name)
				continue
			else:
				if not np.array_equal(weights1[0],weights2[0]):
					# weights differ
					print('weights differ: ',layer1.name)

		except:
			# layer does not exist
			print('layer not found in model2: ',layer1.name)
			pass

def main(**kwargs):
	config = train_inceptionV3_base.load_config(kwargs['config_file_path'])
	model,model_dir,epoch = generateModel(**config)

	# write config pipeline to model dir
	with open(os.path.join(model_dir,'pipeline.yaml'),'w') as f:
		f.write(yaml.dump(config))
	print('[OK] configuration pipeline in model dir')

	epoch_count = 0
	if config['classification_config'].get('enabled',True):
		datagen = image.ImageDataGenerator(
			validation_split=config['classification_config'].get('validation_split',0.0),
			preprocessing_function=tf_preprocessor, # preprocessing function is applied before rescaling
			)
		train_data = datagen.flow_from_directory(
			directory=config['classification_config'].get('data')['train_items_path'],
			target_size=(config['input_shape'][0], config['input_shape'][1]),
			batch_size=config['classification_config'].get('batch_size',train_inceptionV3_base.BATCH_SIZE),
			class_mode='binary',
			shuffle=True,
			#subset='training', # either 'training' or 'validation' of None for all
			#seed=1,
			)
		val_data = datagen.flow_from_directory(
			directory=config['classification_config'].get('data')['val_items_path'],
			target_size=(config['input_shape'][0], config['input_shape'][1]),
			batch_size=config['classification_config'].get('batch_size',train_inceptionV3_base.BATCH_SIZE),
			class_mode='binary',
			shuffle=True,
			subset='training', # either 'training' or 'validation' of None for all
			#seed=1,
			)
		test_data = datagen.flow_from_directory(
			directory=config['classification_config'].get('data')['val_items_path'],
			target_size=(config['input_shape'][0], config['input_shape'][1]),
			batch_size=config['classification_config'].get('batch_size',train_inceptionV3_base.BATCH_SIZE),
			class_mode='binary',
			shuffle=True,
			subset='validation', # either 'training' or 'validation' of None for all
			#seed=1,
			)
		for training_stage in config['classification_config'].get('classification_training',[]):
			stage_epochs = training_stage.get('epochs',30)
			if training_stage.get('enabled',True):

				# freeze
				fine_tune_layer_name = config.get('base_output_layer','')
				if not training_stage.get('freeze_to_bottleneck',False):
					fine_tune_layer_name = training_stage.get('fine_tuning_layer',
						config.get('base_output_layer',''))
				freezeModel(
					model,
					fine_tune_layer_name,
				)

				model.compile(
					optimizer=generateOptimizer(training_stage),
					loss=config['classification_config'].get('loss','categorical_crossentropy'),
					metrics=config['classification_config'].get('metrics',['accuracy']),
				)

				callbacks = generateCallbackList(training_stage,model_dir)

				# prompt model summary
				model.summary()
				print('freeze to {}'.format(fine_tune_layer_name))

				initial_epoch = max(epoch,epoch_count)

				first_training = model.fit_generator(
					generator=train_data,
					steps_per_epoch=len(train_data),
					epochs=epoch_count+stage_epochs,
					verbose=1,
					callbacks=callbacks,
					validation_data=val_data,
					validation_steps=len(val_data),
					initial_epoch=initial_epoch,
				)

				epoch_count = initial_epoch

				try:
					if isinstance(first_training.history['val_loss'],list):
						length = len(first_training.history['val_loss'])
						epoch_count += length
						val_loss = first_training.history['val_loss'][length-1]
						model.save_weights(os.path.join(model_dir,'weights_final__epoch=%02d__val_loss=%.2f.hdf5'%(epoch_count,val_loss)))
				except:
					print('final save failed')
			else:
				epoch_count += stage_epochs

		# final evaluation
		model.compile(
			optimizer='sgd',
			loss=config['classification_config'].get('loss','categorical_crossentropy'),
			metrics=config['classification_config'].get('metrics',['accuracy']),
			)
		predictions = model.evaluate_generator(
			generator=test_data,
			steps=len(test_data),
			verbose=1,
			)
		print(predictions)

if __name__ == '__main__':

	# extract parameters from console input and map along PARAMS
	#options,long_options = console_utils.generate_opt_args_from_lists(PARAMS)
	options,long_options = console_utils.generate_opt_args_from_lists(train_inceptionV3_base.PARAMS)
	opts, args = getopt.getopt(
		sys.argv[1:],
		options,
		long_options,
		)

	# generate dict for extracted opts and args
	kwargs = train_inceptionV3_base.generate_kwargs_dict(opts,args)

	main(**kwargs)
