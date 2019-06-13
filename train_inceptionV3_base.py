import os
import sys
import getopt
import yaml
import re
from utils import extractAnnotations
from utils import console_utils
import train_inceptionV3_classifier
import keras

EXTRACT_ANNOTATIONS_KER = 'extractAnnotations'

CONTROLLER = {
	EXTRACT_ANNOTATIONS_KER:extractAnnotations,
}

PARAMS = [
	[
		("c","path_to_config_file",),
		],
	[
		],
]

CONFIGS_DIR = 'config'
MODELS_DIR = 'models/keras_models'
CONFIG_YAML_FILE = 'default_pipeline.yaml'
MODEL_YAML_FILE = 'model_config.yaml'
IMG_SIZE = 460
BATCH_SIZE = 4
CONFIGS_DIR = 'config'
DEFAULT_CONFIG = {
	'base':'InceptionV3',
	'base_model_ckpt':'weights_init_imagenet.hdf5',
	'base_model_dir':'',
	'base_output_layer':'mixed10',
	'classification_config':{
		'batch_size':BATCH_SIZE,
		'classification_training':[
			{
				'callbacks':{
					'ReduceLROnPlateau':{
						'monitor':'val_loss',
						'factor':0.25,
						'patience':3,
						'verbose':1,
						'mode':'auto',
						#'min_delta':0,
						'cooldown':2,
						'min_lr':0.000025,
						},
					'EarlyStopping':{
						'monitor':'val_loss',
						'min_delta':0.005,
						'patience':10,
						'verbose':1,
						'mode':'auto',
						'baseline':0.75,
						},
					'ModelCheckpoint':{
						'filepath':'weights__epoch={epoch:02d}__val_loss={val_loss:.2f}.hdf5',
						'monitor':'val_loss',
						'verbose':1,
						'save_best_only':False,
						'mode':'auto',
						'save_weights_only':True,
						'period':2,
						},
					'TensorBoard':{
						'log_dir':'logs',
						'write_graph':False,
						#'histogram_freq':2,
						#'write_grads':True,
						'write_images':True,
						'batch_size':BATCH_SIZE,
						},
					},
				'enabled':False,
				'epochs':30,
				'fine_tuning_layer':'mixed10',
				'freeze_to_bottleneck':True,
				'optimizer': {
					'SGD':{
						'lr':0.02,
						'momentum':0.1,
						'decay':0.001,
						'nesterov':False,
						},
					},
				},
			{
				'enabled':False,
				'epochs':30,
				'fine_tuning_layer':'mixed7',
				'freeze_to_bottleneck':False,
				'optimizer': {
					'SGD':{
						'lr':0.0002,
						'momentum':0.9,
						'decay':0.0001,
						'nesterov':False,
						},
					},
				'callbacks':{
					'ReduceLROnPlateau':{
						'monitor':'val_loss',
						'factor':0.25,
						'patience':3,
						'verbose':1,
						'mode':'auto',
						#'min_delta':0,
						'cooldown':2,
						'min_lr':0.000025,
						},
					'EarlyStopping':{
						'monitor':'val_loss',
						'min_delta':0.005,
						'patience':10,
						'verbose':1,
						'mode':'auto',
						'baseline':0.75,
						},
					'ModelCheckpoint':{
						'filepath':'weights__epoch={epoch:02d}__val_loss={val_loss:.2f}.hdf5',
						'monitor':'val_loss',
						'verbose':1,
						'save_best_only':False,
						'mode':'auto',
						'save_weights_only':True,
						'period':2,
						},
					'TensorBoard':{
						'log_dir':'logs',
						'write_graph':False,
						#'histogram_freq':2,
						#'write_grads':True,
						'write_images':True,
						'batch_size':BATCH_SIZE,
						},
					},
				},
			],
		'data':{
			'train_items_path':os.path.join('data','images','train_classification_dataset','train_min'),
			'val_items_path':os.path.join('data','images','train_classification_dataset','test_min'),
			},
		'enabled':True,
		'head_layers':{
			'averagePooling2D':1,
			'conv2d':[
				(256,1,1,'valid','relu',{'alpha':0.15}),
				(256,3,2,'valid','relu',{'alpha':0.15},{'name':'bottleneck_feats','padding':1}),
				],
			'dense':[
				(256,'relu',{'alpha':0.15}),
				(128,'relu',{'alpha':0.15}),
				(1,'sigmoid'),
				],
			'dropout':[
				0.5,
				0.5,
				],
			'maxPooling2D':1,
			},
		'loss':'binary_crossentropy',
		'metrics':[
					'accuracy',
					],
		'validation_split':0.3,
		'weight_checkpoint_file':'weights_init_imagenet.hdf5'
	},
	'detection_config':{
		'alpha':[0.5,1.0,1.5,2.0,2.5],
		'aspect_ratios':[1.,2.,3.,1./2.,1./3.],
		'batch_size':BATCH_SIZE,
		'data_path':'data/trainings/graffiti_ba_training/annotations',
		'default_kernel_size':3,
		'enabled':True,
		'feature_map_filter_sizes':512,
		'feature_pyramid_network':False,
		'fpn_filter_sizes':256,
		'loss':['combined_loss'],
		'max_anchor_limit':.98,
		'max_scale':.9,
		'min_kernel_size':3,
		'min_scale':.1,
		'neg_iou_assign_threshold':0.4,
		'parameter_sharing':False,
		'pos_iou_assign_threshold':0.6,
		'sigma':2.0,
		'similarity_func':'jaccard_similarity',
		'spatial_drop_out_rate':0.5,
		'subdir_name':'defaultKerasDetectionModel',
		'test_split_pattern':'test',
		'train_test_split':True,
		'trainings':[
			{
				'callbacks':{
					'ModelCheckpoint':{
						'filepath':'0_weights__epoch={epoch:02d}__loss={loss:.2f}__val_loss={val_loss:.2f}.hdf5',
        				'mode':'auto',
						'monitor':'val_loss',
						'period':15,
						'save_best_only':True,
						'save_weights_only':True,
        				'verbose':1
						},
					'ReduceLROnPlateau':{
						'cooldown':20,
						'factor':0.5,
						'min_lr':2.5e-05,
						'mode':'auto',
						'monitor':'val_loss',
						'patience':15,
 						'verbose':1
						},
					'TensorBoard':{
						'batch_size':4,
						'log_dir':'logs',
						'write_graph':False,
						'write_images':True
						},
					},
				'enabled':False,
				'epochs':50,
				'fine_tuning_layer':'mixed10',
				'freeze_to_bottleneck':False,
				'optimizer':{
					'SGD':{
						'decay':0.0005,
 						'lr':0.005,
						'momentum':0.4,
 						'nesterov':False
						},
					},
				},
			{
				'callbacks':{
					'ModelCheckpoint':{
						'filepath':'0_weights__epoch={epoch:02d}__loss={loss:.2f}__val_loss={val_loss:.2f}.hdf5',
        				'mode':'auto',
						'monitor':'val_loss',
						'period':15,
						'save_best_only':True,
						'save_weights_only':True,
        				'verbose':1
						},
					'ReduceLROnPlateau':{
						'cooldown':20,
						'factor':0.5,
						'min_lr':2.5e-05,
						'mode':'auto',
						'monitor':'val_loss',
						'patience':15,
 						'verbose':1
						},
					'TensorBoard':{
						'batch_size':4,
						'log_dir':'logs',
						'write_graph':False,
						'write_images':True
						},
					},
				'enabled':False,
				'epochs':40,
				'fine_tuning_layer':'mixed7',
				'freeze_to_bottleneck':False,
				'optimizer':{
					'SGD':{
						'decay':0.0005,
 						'lr':0.0025,
						'momentum':0.9,
 						'nesterov':False
						},
					},
				},
			],
		'weight_checkpoint_file':'weights_init_classification_00.hdf5',
		},
	'feature_map_layers':[
		'mixed7',
		'mixed10',
		],
	'input_shape':(IMG_SIZE,IMG_SIZE,3),
}

def generate_kwargs_dict(opts,args):
	'''
	Generates parameter dictionary for internal use from provided opts and args typles.
	Checks presence for each short/long option pair provided by PARAMS in the opts list
	and extracts the provided value	if option is found.

	# Args:
		opts: list of of ([long]name, value) pairs holding, names could be included in PARAMS
		args: list of additional args provided through execution call

	# Returns:
		kwargs: dictionary holding key value pairs, where
		  keys are the names of internal function parameters and
		  values are the corresponding arguments that should be passed in the function call
			- config_file_path: path to a valid configuration yaml file
	'''
	kwargs = {}

	config_file_path = console_utils.extract_option(PARAMS[0][0],opts)
	if config_file_path is None:
		config_file_path = CONFIG_YAML_FILE
	config_file_path = os.path.splitext(config_file_path)[0]+'.yaml'
	if os.path.isfile(config_file_path):
		kwargs['config_file_path'] = config_file_path
	else:
		kwargs['config_file_path'] = os.path.join(CONFIGS_DIR,config_file_path.split('/')[-1])

	return kwargs

def generateOptimizer(optimizer,**kwargs):
	for className,kwargs in optimizer.items():
		optimizer = getattr(keras.optimizers,className)
		return optimizer(**kwargs)

def generateCallbackList(callbacks,model_dir,loss_function_name,alpha_value,**kwargs):
	fit_callbacks = []
	loss_config = '{}__alpha={:.2f}'.format(loss_function_name,alpha_value)
	for className,params in callbacks.items():
		callback_args = params.copy()
		callback = getattr(keras.callbacks,className)
		if callback is keras.callbacks.TensorBoard:
			callback_args['log_dir']=os.path.join(model_dir,callback_args['log_dir'],loss_config)
		elif callback is keras.callbacks.ModelCheckpoint:
			file_prefix,file_extension = os.path.splitext(callback_args['filepath'])
			file_name = '{}__{}{}'.format(loss_config,file_prefix,file_extension)
			callback_args['filepath']=os.path.join(model_dir,file_name)
		fit_callbacks.append(callback(**callback_args))
	return fit_callbacks

def model_from_yaml(model_dir,custom_objects={}):
	''' Loads a model from a configuration file model_dir/model_config.yaml
	Returns:
		model
	'''
	with open(os.path.join(model_dir,MODEL_YAML_FILE),'r') as f:
		yaml_string = f.read()
	return keras.models.model_from_yaml(yaml_string,custom_objects)

def write_model_yaml(model, model_dir):
	''' Writes model configuration to model_config.yaml file and saves
	to file into model_dir.
	'''
	with open(os.path.join(model_dir,MODEL_YAML_FILE),'w') as f:
		f.write(model.to_yaml())

def getModelDir_decorate(model_subdir_func,config):
	'''
	Returns:
		wrapper function handle to create model base directory
			and manage subdir names according to pipeline setup.
	'''
	def wrapper(**kwargs):
		base_dir = base_directory(**config)
		model_dir = os.path.join(base_dir,*(model_subdir_func(**kwargs)))
		if not os.path.exists(model_dir):
			os.makedirs(model_dir)
		return model_dir
	return wrapper

def base_directory(base,base_output_layer='',input_shape=(299,299,3),feature_map_layers=[],**kwargs):
	''' Generates a model base directory name.
	All classification and detection models are stored in this folder.
	The model base directory is generated from the config field:
	base, base_output_layer, input_shape, feature_map_layers

	Returns:
		relative path to the model dir (located under ./models/keras_models)
	'''
	s = ''
	for feature_layer in feature_map_layers:
		s += '{},'.format(feature_layer)
	s = s[:-1]
	directory_name = '{}_{}_input={}x{}_featMap={}'.format(base,base_output_layer,input_shape[0],input_shape[1],s)
	return os.path.join(MODELS_DIR,directory_name)

def _fix_config(config):
	drop_rates = []
	for layer in range(len(config['dense'])):
		if isinstance(config['dropout'],list):
			try:
				rate = config['dropout'][layer]
			except:
				rate = 0.0
		else:
			rate = config['dropout']
		assert isinstance(rate,float), 'dropout rate must be of type <float>'
		drop_rates.append(rate)

	assert len(config['dense']) == len(drop_rates)
	config['dropout'] = drop_rates

	#base_output_layer = config.get('base_output_layer','')

def load_config(config_file_path,**kwargs):
	try:
		with open(os.path.join(config_file_path),'r') as setting:
			config = yaml.load(setting.read())
		print('use provided config')
	except:
		print('use default config')
		config = DEFAULT_CONFIG

		# write config file
		confFilePath = os.path.join(
			CONFIGS_DIR,
			CONFIG_YAML_FILE)

		if not os.path.exists(confFilePath):
			with open(confFilePath,'w') as f:
				f.write(yaml.dump(config))
	finally:
		_fix_config(config['classification_config']['head_layers'])
	return config

def list_model_weights(model_dir):
	pattern = '.*\.hdf5'
	files = os.listdir(model_dir)
	weights = []
	for file in files:
		m = re.match(re.compile(pattern),file)
		if m is not None:
			weights.append(file)
	return weights

def list_tf_checkpoints(model_dir):
	pattern = 'ckpt-(\d+)'
	weights = []
	queue = ['']
	while len(queue) > 0:
		subdir_path = queue.pop()
		directory = os.path.join(model_dir,subdir_path)
		items = os.listdir(directory)
		for item in items:
			item_path = os.path.join(directory,item)
			if os.path.isdir(item_path):
				m = re.match(re.compile(pattern),item)
				if m is not None:
					weights.append(os.path.join(subdir_path,item,'frozen_inference_graph.pb'))
				else:
					queue.append(os.path.join(subdir_path,item))
	return weights
