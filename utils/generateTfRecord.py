'''
# Execution examples (call from Graffiti_Detection/ directory):

python utils/generateTfRecord.py \
	--path_to_training_directory=data/trainings/graffiti_ba_training/ \
	--test_item_pattern=test

python utils/generateTfRecord.py \
	--path_to_annotations_root=data/trainings/graffiti_ba_training/annotations/ \
	--path_to_training_directory=data/trainings/graffiti_ba_training/ \
	--test_item_pattern=test \
	--max_chunk_size=1000

# Args:
	-i --path_to_annotations_root directory from where to start searching for annotation files
	-o --path_to_training_directory directory (will be created if not existing) where to store the generated trainig data
	-t --test_item_pattern name of a directory where the annotations for the test data is included
	-c --max_chunk_size max number of items per tf record chunk
	-l --path_to_label_map label map for which the tf records should be generated

	-s --train_test_split flag only parameter indicates that extracted annotation data should be split into test and train data
	-u --update_provided_label_map flag only parameter indicates the a given label map will be updated if new labels occur in the data
'''

import sys
import os
import math
import hashlib
import json
import yaml
import getopt
import tensorflow as tf
from lxml import etree
import contextlib2
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import label_map_util

from utils import extractAnnotations, console_utils

PARAMS = [
	[
		("i","path_to_annotations_root",),
		("o","path_to_training_directory",),
		("t","test_item_pattern",),
		("c","max_chunk_size",),
		("l","path_to_label_map",),
		],
	[
		('s','train_test_split',),
		('u','update_provided_label_map',),
		],
]

TFRECORD_SUBDIR = 'TFRecords'
INFO_FILE_NAME = 'data_info.yaml'
LABEL_MAP_NAME = 'label_map.pbtxt'
MAX_CHUNKS_SIZE = 1000

def _get_label_map_path(training_dir):
	return os.path.join(training_dir,LABEL_MAP_NAME)

def _get_info_file_path(training_dir):
	return os.path.join(training_dir,INFO_FILE_NAME)

def _write_label_map_file(label_map_path,label_map_dict):
	'''Writes a tensorflow object api compatible label map proto file to disk.

	# Args:
		label_map_dict dictionary containing the mapping from class names to class ids
		trainig_dir path to the directory where the file should be written to
	'''
	if not label_map_path:
		print('skip generation of label map proto file')
		return

	label_map_string = ''
	for object_slug,object_id in label_map_dict.items():
		item_string = 'item {\n'
		item_string += '\tid: {:d}\n'.format(object_id)
		item_string += '\tname: \'{}\'\n'.format(object_slug)
		item_string += '\tdisplay_name: \'{}\'\n'.format(object_slug)
		item_string += '}\n\n'
		label_map_string += item_string

	with open(label_map_path,'w') as f:
		f.write(label_map_string)

def _write_info_file(info_file_path,data,label_map_dict,training_dir,data_dir,**kwargs):
	'''Writes an info file into the trainig directory.
	'''
	data_fingerprint = hashlib.sha256(json.dumps(data).encode('utf-8')).hexdigest()
	class_fingerprint = hashlib.sha256(json.dumps(label_map_dict).encode('utf-8')).hexdigest()
	num_train_items = len(data['train'])
	num_test_items = len(data['test'])
	data_folder_list = extractAnnotations.get_data_dir_list(data)

	info = {
		'image root directory':data_dir,
		'number of items': [num_test_items+num_train_items,{'test': num_test_items, 'train': num_train_items}],
		'image data fingerprint':data_fingerprint,
		'label map fingerprint':class_fingerprint,
		'image data directories':data_folder_list,
		'test directory pattern':kwargs['test_dir_pattern'],
		}

	with open(info_file_path,'w') as f:
		f.write(yaml.dump(info))

def _create_tf_records(data,label_map_dict,training_dir,max_chunks_size,**kwargs):
	'''Creates the tf record chunks from the provided data and label map.
	'''
	for mode,items in data.items():
		output_filebase='{}/{}_dataset.record'.format(
			training_dir,
			mode)
		num_chunks = math.ceil(len(items) / max_chunks_size)
		if num_chunks > 0:
			with contextlib2.ExitStack() as tf_record_close_stack:
				output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
					tf_record_close_stack,
					output_filebase,
					num_chunks)
				i = 0
				for item in items:
					try:
						tf_example = extractAnnotations.create_tf_example(
							item,
							label_map_dict)
						if tf_example is not None:
							output_shard_index = i % num_chunks
							output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
							i += 1
						else:
							#print(item['annotation']['path'])
							#raise Exception('no bounding box annotations found for item {}'.format(item['annotation']['path']))
							pass
					except Exception as e:
						print(e)
						print('could not generate tf example for item {}'.format(item['annotation']['path']))

def generate_training_data(data,label_map_dict,label_map_path,info_file_path,**kwarg):
	'''File generation pipeline
	'''
	_create_tf_records(data,label_map_dict,**kwargs)
	_write_label_map_file(label_map_path,label_map_dict)
	_write_info_file(info_file_path,data,label_map_dict,**kwargs)

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
	'''
	kwargs = {}


	training_path_param = console_utils.extract_option(PARAMS[0][1],opts)
	if training_path_param is None:
		raise EnvironmentError('no training data path defined')
	kwargs['training_dir'] = os.path.abspath(os.path.join(training_path_param,TFRECORD_SUBDIR))

	root = console_utils.extract_option(PARAMS[0][0],opts)
	if root is None:
		kwargs['data_dir'] = os.path.abspath(os.path.join(training_path_param,extractAnnotations.TRAINING_ANNOTATIONS_SUBDIR))
	else:
		kwargs['data_dir'] = os.path.abspath(root)

	if not os.path.isdir(kwargs['data_dir']):
		raise EnvironmentError('no root path defined')

	test_split_flag = True
	test_dir_pattern = console_utils.extract_option(PARAMS[0][2],opts)
	if test_dir_pattern is None:
		test_split_flag = console_utils.check_flag_presence(PARAMS[1][0],opts)
		test_dir_pattern = extractAnnotations.TEST_DATA_KEY
	kwargs['test_dir_pattern'] = test_dir_pattern
	kwargs['train_test_split'] = test_split_flag

	chunk_size = console_utils.extract_option(PARAMS[0][3],opts)
	if chunk_size is None:
		chunk_size = MAX_CHUNKS_SIZE
	kwargs['max_chunks_size'] = int(chunk_size)

	path_to_label_map = console_utils.extract_option(PARAMS[0][4],opts)
	update_label_map = False
	if path_to_label_map is not None:
		kwargs['path_to_label_map'] = path_to_label_map
		update_label_map = console_utils.check_flag_presence(PARAMS[1][1],opts)
	kwargs['update_label_map'] = update_label_map

	return kwargs

def main(**kwargs):
	'''Performs parameter extraction/mapping and processing.
	'''
	training_dir = kwargs.get('training_dir',None)
	if not training_dir:
		raise Exception

	root_dir = kwargs.get('data_dir',None)

	data = extractAnnotations.process(data_path=root_dir, bb_encoding_0_1=True, **kwargs)

	info_file_path = _get_info_file_path(training_dir)

	label_map_path = _get_label_map_path(training_dir)
	provided_label_map_path = kwargs.get('path_to_label_map',None)
	if provided_label_map_path:
		label_map_dict = label_map_util.get_label_map_dict(provided_label_map_path)
		update_label_map = kwargs.get('update_label_map',False)
		if update_label_map:
			data_label_map_dict = extractAnnotations.generate_label_map_from_data(data)
			next_class_index = len(label_map_dict)+1
			for data_class_name,_ in data_label_map_dict.items():
				if data_class_name not in label_map_dict.keys():
					label_map_dict[data_class_name] = next_class_index
					next_class_index +=1
		elif os.path.abspath(provided_label_map_path) == os.path.abspath(label_map_path):
			label_map_path = None
	else:
		label_map_dict = extractAnnotations.generate_label_map_from_data(data)

	try:
		os.makedirs(training_dir)
		generate_training_data(data,label_map_dict,label_map_path,info_file_path,**kwargs)
	except:
		# dir exists already
		info_file_exists = os.path.exists(info_file_path) and os.path.isfile(info_file_path)
		if info_file_exists:
			with open(info_file_path,'r') as f:
				info = yaml.load(f.read())
		else:
			_write_info_file(info_file_path,data,label_map_dict,**kwargs)
		generate_training_data(data,label_map_dict,label_map_path,info_file_path,**kwargs)

if __name__ == '__main__':

	# extract parameters from console input and map along PARAMS
	options,long_options = console_utils.generate_opt_args_from_lists(PARAMS)
	opts, args = getopt.getopt(
		sys.argv[1:],
		options,
		long_options,
		)

	# generate dict for extracted opts and args
	kwargs = generate_kwargs_dict(opts,args)

	main(**kwargs)
