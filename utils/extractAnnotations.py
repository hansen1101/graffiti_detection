'''
# Execution examples (call from Graffiti_Detection/ directory):
python utils/extractAnnotations.py \
	-r \
	-m data/example_data/images_and_annotations/ \
	-t datasource_1 \
	-e images_and_annotations

python utils/extractAnnotations.py \
	-rxs \
	-m data/example_data/images_and_annotations/

python utils/extractAnnotations.py \
	--markup_root=data/example_data/images_and_annotations/ \
	--relative \
	--export \
	--split \

python utils/extractAnnotations.py \
	--markup_root=data/example_data/annotations_only/ \
	--training_name=example_training \
	--test_fraction=0.4 \
	--image_root_dir=data/example_data/images_and_annotations/ \
	-r


# Args:
	-m --markup_root directory from where to start searching for annotation files
	-e --export_file name prefix of a json file to which the extracted annotations should be written
	-t --test_dir_name name of a directory where the annotations for the test data is included
	-n --training_name name of the training process
	-f --test_fraction fraction of the data that are included in the validation/test dataset
	-o --training_dir root directory where the training data should be put into
	-i --image_root_dir directory where the images are located if the data is moved

	-r --relative flag only parameter indicates that bounding boxes are encoded relative (0.,1.) towards original image size
	-x --export flag only parameter indicates that extracted annotation data should be exported as a json file
	-s --split flag only parameter indicates that extracted annotation data should be split into test and train data
	-c --coco
'''

import os
import sys
import io
import getopt
import json
import hashlib
import shutil
import random
import xml.etree.ElementTree as ET
import tensorflow as tf
import PIL
import math
from object_detection.utils import dataset_util
from pycocotools import coco

from utils import console_utils

PARAMS = [
	[
		("m","markup_root",),
		("e","export_file",),
		("t","test_dir_name",),
		("n","training_name",),
		("f","test_fraction",),
		("o","training_dir",),
		("i","image_root_dir",),
		],
	[
		('r','relative',),
		('x','export',),
		('s','split',),
		('c','coco',),
		],
]

EXPORT_DIR = os.path.join('data','exported_annotations')
EXPORT_FILE_NAME = 'annotations.json'

TRAINING_PROCESSES_ROOT = 'data/trainings'
TRAINING_ANNOTATIONS_SUBDIR = 'annotations'
TRAINING_DATA_KEY = 'train'
TEST_DATA_KEY = 'test'

def generate_kwargs_dict(opts,args):
	'''
	Generates parameter dictionary for internal use from provided opts and args typles.
	Checks presence for each short/long option pair provided by PARAMS in the opts list
	and extracts the provided value	if option is found.

	Args:
		opts: list of of ([long]name, value) pairs holding, names could be included in PARAMS
		args: list of additional args provided through execution call

	Returns:
		kwargs: dictionary holding key value pairs, where
		  keys are the names of internal function parameters and
		  values are the corresponding arguments that should be passed in the function call
			- data_path: name of the root directory where the xml bounding box
				annotation makups live
			- bb_encoding_0_1: boolean indicates whether the bounding box encodings
				should be converted into relative encodings
			- test_dir_pattern: name of the directory where the test data live
			- train_test_split: boolean indicates whether the data should be
				split into two distinct datasets
			- export_file_name: name of the file where the data should be written
				to if the annotation dict should be exported to a json file
			- export_flag: boolean indicates whether the data should be exported
				to a json file
			- coco_flag: boolean indicates whether the annotations are coco json file
			- training_name: name of the directory in which the data for the
				training process is organized
			- test_frac: fraction of the data that should be reserved for the
				test/validation dataset (value between 0 and 1 amount)
			- training_root_dir: [default: TRAINING_PROCESSES_ROOT] root directory
				where training processes are stored
			- image_root_path: root path to the images
	'''
	kwargs = {}

	markup_root = console_utils.extract_option(PARAMS[0][0],opts)
	if markup_root is None:
		raise EnvironmentError('no markup_root path defined')
	else:
		kwargs['data_path'] = markup_root

	kwargs['bb_encoding_0_1'] = console_utils.check_flag_presence(PARAMS[1][0],opts)

	test_split_flag = True
	test_dir_pattern = console_utils.extract_option(PARAMS[0][2],opts)
	if test_dir_pattern is None:
		test_split_flag = console_utils.check_flag_presence(PARAMS[1][2],opts)
		test_dir_pattern = TEST_DATA_KEY
	kwargs['test_dir_pattern'] = test_dir_pattern
	kwargs['train_test_split'] = test_split_flag

	export_flag = True
	export_file_name = console_utils.extract_option(PARAMS[0][1],opts)
	if export_file_name is None:
		export_flag = console_utils.check_flag_presence(PARAMS[1][1],opts)
		export_file_name = EXPORT_FILE_NAME
	# ensure export file has .json extension
	_,export_file_ext = os.path.splitext(export_file_name)
	if export_file_ext != '.json':
		export_file_name += '.json'
	kwargs['export_file_name'] = export_file_name
	kwargs['export_flag'] = export_flag

	kwargs['coco_flag'] = console_utils.check_flag_presence(PARAMS[1][3],opts)

	kwargs['training_name'] = console_utils.extract_option(PARAMS[0][3],opts)
	test_frac = console_utils.extract_option(PARAMS[0][4],opts)
	if test_frac is not None:
		kwargs['test_frac'] = float(test_frac)

	kwargs['training_root_dir'] = TRAINING_PROCESSES_ROOT
	training_root_dir = console_utils.extract_option(PARAMS[0][5],opts)
	if training_root_dir is not None:
		kwargs['training_root_dir'] = training_root_dir

	kwargs['image_root_path'] = kwargs['data_path']
	image_root_path = console_utils.extract_option(PARAMS[0][6],opts)
	if image_root_path is not None:
		if not os.path.isdir(image_root_path):
			print('Warning: image root path {} is not a valid directory > use {} instead'.format(image_root_path,os.getcwd()))
			image_root_path = os.getcwd()
		kwargs['image_root_path'] = image_root_path

	return kwargs

def _recursive_parse_xml_to_dict(node,filePrefix,update_xml=False,image_root_path=None,**kwargs):
	'''
	Performs the data extraction on a node of the xml tree in a recursice fashion.
	Updates the node directly if required and signals any update by passing a flag back to the callee.

	# Args:
		node the xml node which is processed
		filePrefix full path to the markup file without file extension

	# Returns:
		dictionary with single node tag and node value pair for current node
		saveElementChanges bool value indicating any changes performed on this or any
		  child node
	'''
	saveElementChanges = False
	if not node:
		# base case: node contains element
		data = node.text
		if node.tag == 'folder':
			if update_xml:
				dir_fractions = filePrefix.split('/')
				item_folder = dir_fractions[len(dir_fractions)-2]
				try:
					assert item_folder == node.text
				except:
					node.text = item_folder
					saveElementChanges = True
			data = node.text
		elif node.tag == 'path':
			img_path = os.path.abspath(node.text)
			if not os.path.isfile(img_path) and image_root_path is not None:
				print('{:>9} updating path with base...'.format('...'))
				path_fractions = img_path.split('/')
				found = False
				print(image_root_path)
				for i in range(len(path_fractions)-1,-1,-1):
					for j in range(i,len(path_fractions)):
						img_path_candidate = os.path.abspath(os.path.join(image_root_path,*path_fractions[j:]))
						#print(img_path_candidate)
						if os.path.isfile(img_path_candidate):
							# found
							print('{:>9} set value\n{:>9} from:\t{}\n{:>9} to:\t{}\n'.format('...','',node.text,'',img_path_candidate))
							node.text = img_path_candidate
							saveElementChanges = True
							found = True
							break
					if found:
						break
				if not found:
					print('Warning: could not update invalid img path {}'.format(img_path))
			if update_xml:
				imgItemParts = os.path.splitext(node.text)
				try:
					assert imgItemParts[0] == filePrefix
				except:
					node.text = filePrefix+imgItemParts[1]
					saveElementChanges = True
			data = node.text
		elif node.tag in ['width','height','depth','xmin','xmax','ymin','ymax']:
			data = int(node.text)

		return {node.tag: data},saveElementChanges

	item = dict()
	for child in node:
		data,changeSignal = _recursive_parse_xml_to_dict(child,filePrefix,update_xml,image_root_path,**kwargs)
		if child.tag == 'object':
			if child.tag not in item:
				item[child.tag] = []
			item[child.tag].append(data[child.tag])
		else:
			item[child.tag] = data[child.tag]

		if changeSignal:
			saveElementChanges = True

	return {node.tag: item},saveElementChanges

def _generate_coco_data_dict(imgs=[],annos=[],cats=[]):
	img = imgs[0]
	annotation = {
		'folder':'',
		'filename':img['file_name'],
		'path':img['coco_url'],
		'source':{'database':'COCO'},
		'size':{'width':img['width'],'height':img['height'],'depth':3},
		'segmented':0,
		'object':[],
	}
	assert len(annos) == len(cats)
	for i,anno_obj in enumerate(annos):
		category = cats[i]
		xmin = round(anno_obj['bbox'][0])
		ymin = round(anno_obj['bbox'][1])
		xmax = round(anno_obj['bbox'][0]+round(anno_obj['bbox'][2]))
		ymax = round(anno_obj['bbox'][1]+round(anno_obj['bbox'][3]))

		try:
			math.log(xmax-xmin)
			math.log(ymax-ymin)
		except Exception as e:
			continue
		annotation['object'].append({
			'name':category['name'],
			'class_id':category['id'],
			'super_class':category['supercategory'],
			'pose':'',
			'truncated':0,
			'difficult':0,
			'bndbox':{
				'xmin':xmin,
				'ymin':ymin,
				'xmax':xmax,
				'ymax':ymax}
		})

	return {'annotation':annotation}

def _process_data(data,bb_encoding_0_1):
	'''
	Adds some additional fields that are not covered by the xml markup file structure
	to the data dictionary. Performs value conversion from absolute to relative values.

	# Args:
		data dictionary containing the item's annotated data
		bb_encoding_0_1 bool value indicating if bounding box encoding should be
		  relative (True) or absolute (False)
	'''
	data = data['annotation']
	width = data['size']['width']
	height = data['size']['height']

	# add width/height/center info for any bounding box
	if 'object' in data:
		for bbox in data['object']:
			bbox['bndbox']['bwidth'] = bbox['bndbox']['xmax'] - bbox['bndbox']['xmin']
			bbox['bndbox']['bheight'] = bbox['bndbox']['ymax'] - bbox['bndbox']['ymin']
			bbox['bndbox']['xcenter'] = bbox['bndbox']['xmin'] + bbox['bndbox']['bwidth'] // 2
			bbox['bndbox']['ycenter'] = bbox['bndbox']['ymin'] + bbox['bndbox']['bheight'] // 2

		# convert all bboxes
		if bb_encoding_0_1:
			for bbox in data['object']:
				bbox['bndbox']['xmin'] /= width
				bbox['bndbox']['xmax'] /= width
				bbox['bndbox']['bwidth'] /= width
				bbox['bndbox']['xcenter'] /= width
				bbox['bndbox']['ymin'] /= height
				bbox['bndbox']['ymax'] /= height
				bbox['bndbox']['bheight'] /= height
				bbox['bndbox']['ycenter'] /= height

def _extract_from(xml_file_path,bb_encoding_0_1,**kwargs):
	'''
	Extracts the annotations for a markup file. Keeps the parse tree data
	updated according to the signal returned by the underlying recrusion.
	Extracted data is also processed in order to meet the given bb_encoding_0_1 flag.

	# Args:
		xml_file_path: path to a markup file
		bb_encoding_0_1: bool value indicating if the extracted bounding boxes are encoded
		  in relative values (True) towards the image width/height or absolute values (False)

	# Returns:
		data dictionary matching the underlying markup file structure
	'''
	xmlItemPrefix,xmlItemPostfix = os.path.splitext(xml_file_path)
	tree = ET.parse(xml_file_path)
	root = tree.getroot()
	data,save = _recursive_parse_xml_to_dict(root,xmlItemPrefix,**kwargs) # start recusrion at root
	if save:
		tree.write(open(xml_file_path, 'wb'))
	_process_data(data,bb_encoding_0_1)
	return data

def _get_annotation_files(data_path,test_dir_pattern='test'):
	'''
	Searches data_path for xml markup files and generates a list of
	full paths off all found xml files to the annotationFiles lists.
	Keeps two separate lists for train and test sets and assigns each items
	according to the provided test_dir_pattern.
	Search is performed recursively in a top down fashion.
	Args:
		data_path: the point from where to start the search for markup files
		test_dir_pattern: pattern which is matched against an item's full path to
		  identify test items
	Returns:
		annotationFiles: list [2] with two lists containing full paths to markup files
			- The first list [num_train_items] contains all markup files that are
			included in directories not matching the test_dir_pattern. This data
			is considered to be used for the training step.
			- The second list [num_test_items] contains all markup files that
			are included in directories matching the test_dir_pattern. This
			data is considered to be used in test/validation step.
	'''
	annotationFiles = list([list(),list()])
	if not os.path.isdir(data_path):
		print('<data_path> is either a file or does not exist.')
	else:
		pathTree = os.walk(data_path)
		for root, dirs, files in pathTree:
			for file in files:
				fileParts = os.path.splitext(file)
				if fileParts[len(fileParts)-1] == '.xml':
					# xml file is found
					filePath = os.path.join(root,file)
					test = 0
					if isinstance(test_dir_pattern,str):
						if filePath.find(test_dir_pattern) != -1:
							# test_dir_pattern is found in the file path
							test = 1
					annotationFiles[test].append(filePath)
	return annotationFiles

def get_data_dir_list(data):
	'''Generates list of image data directories for training and test items.
	'''
	train_dirs = []
	test_dirs = []
	for mode,items in data.items():
		if mode == TRAINING_DATA_KEY:
			dirs = train_dirs
		else:
			dirs = test_dirs
		for item in items:
			path_fractions = item['annotation']['path'].split('/')
			item_dir = os.path.join(*path_fractions[:len(path_fractions)-1])
			if item_dir not in dirs:
				dirs.append(item_dir)
	return {TRAINING_DATA_KEY:train_dirs,TEST_DATA_KEY:test_dirs}

def generate_label_map_from_data(data):
	'''Generates a mapping from class names to class ids.

	All class names are sorted before id assignment. Successive calls
	with changes in the data set could yield different class id assignments.

	# Args:
		data: dictionary with lists containing extracted annotations

	# Return:
		label_map: dictionary mapping class names to ids
	'''
	labels = []
	for mode,items in data.items():
		for item in items:
			annotation = item['annotation']
			if 'object' in annotation:
				boxes = annotation['object']
				for box in boxes:
					label = box['name']
					if label not in labels:
						labels.append(label)
	label_map = dict()
	for i,label in enumerate(sorted(labels)):
		label_map[label] = i+1
	return label_map

def create_tf_example(item,label_map_dict,ignore_difficult_instances=False):
	''' Generates a tf example from an item with respect to a label map.
	Takes an item's data dictionary and a label map dictionaty and generates tf examples
	that can be used by the tensorflow object detection api (e.g. to generate tfRecord files).
	Only includes an object's bounding box if the class name is present in the label map.

	# Args:
		item dictionary containing the extracted annotations of a single item
		label_map_dict dictionary containing a map from class names to the set of natural number
		ignore_difficult_instances bool value indicating that difficult labeled bounding boxes should
		  be ignored (True) in the examples or not (False)

	# Returns:
		tf.train.example or None if no object from label map is present in the item's dataset.

	'''
	data = item['annotation']
	height = data['size']['height'] # Image height
	width = data['size']['width'] # Image width
	source_id = os.path.splitext(data['filename'])[0] # Image width

	encoded_image_data = None # Encoded image bytes
	filepath = data['path'] # Filename of the image. Empty if image is not from file

	# read image bytes
	with tf.gfile.GFile(filepath, 'rb') as fid:
		encoded_image_data = fid.read()

	encoded_jpg_io = io.BytesIO(encoded_image_data)
	image = PIL.Image.open(encoded_jpg_io)
	image_format = image.format#.encode('utf8')

	key = hashlib.sha256(encoded_image_data).hexdigest()

	difficult_obj = []
	xmin = []
	ymin = []
	xmax = []
	ymax = []
	xcenter = []
	ycenter = []
	bwidth = []
	bheight = []
	classes_text = []
	classes = []
	truncated = []
	poses = []

	if 'object' in data:
		for obj in data['object']:
			difficult = bool(int(obj['difficult']))
			if ignore_difficult_instances and difficult:
				continue

			class_index = label_map_dict.get(obj['name'],None)
			if class_index is None:
				continue

			difficult_obj.append(int(difficult))
			xmin.append(obj['bndbox']['xmin'])
			ymin.append(obj['bndbox']['ymin'])
			xmax.append(obj['bndbox']['xmax'])
			ymax.append(obj['bndbox']['ymax'])
			xcenter.append(obj['bndbox']['xcenter'])
			ycenter.append(obj['bndbox']['ycenter'])
			bwidth.append(obj['bndbox']['bwidth'])
			bheight.append(obj['bndbox']['bheight'])
			classes_text.append(obj['name'].encode('utf8'))
			classes.append(class_index)
			truncated.append(int(obj['truncated']))
			poses.append(obj['pose'].encode('utf8'))

	if len(xmin) == 0:
		# no bounding box for label map found
		return None

	example = tf.train.Example(
		features=tf.train.Features(
			feature={
				'image/height': dataset_util.int64_feature(height),
				'image/width': dataset_util.int64_feature(width),
				'image/filename': dataset_util.bytes_feature(
					data['filename'].encode('utf8')),
				'image/source_id': dataset_util.bytes_feature(source_id.encode('utf8')),
				'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
				'image/encoded': dataset_util.bytes_feature(encoded_image_data),
				'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
				'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
				'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
				'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
				'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
				'image/object/bbox/xcenter': dataset_util.float_list_feature(xcenter),
				'image/object/bbox/ycenter': dataset_util.float_list_feature(ycenter),
				'image/object/bbox/bwidth': dataset_util.float_list_feature(bwidth),
				'image/object/bbox/bheight': dataset_util.float_list_feature(bheight),
				'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
				'image/object/class/label': dataset_util.int64_list_feature(classes),
				'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
				'image/object/truncated': dataset_util.int64_list_feature(truncated),
				'image/object/view': dataset_util.bytes_list_feature(poses),
				}
			)
		)
	return example

def _process(data_path,bb_encoding_0_1=False,train_test_split=False,test_dir_pattern='test',**kwargs):
	'''
	Extracts annotations starting from a root directory. The data is split
	into test and train sets.

	# Args:
		data_path: directory that serves as starting point for tree walk
		bb_encoding_0_1: boolean indicates whether bounding box anchor points are encoded
		  relative (True) or absolute (False) towards image seize
		train_test_split: boolean indicates whether annotations should be splitted into
		  train/test sets (True) or should be handled as only training data (False)
		test_dir_pattern: directory slug that serves as indicator for test data
		kwargs: optional keyword arguments are irgnored

	# Returns:
		data dictionary containing two key value pairs, key indicates whether annotations in the
		  value's list represent the test or train set.
	'''

	data = dict({
		TRAINING_DATA_KEY:list(),
		TEST_DATA_KEY:list(),
		})

	if train_test_split and test_dir_pattern is not None:
		test_dir_pattern = '/{}/'.format(test_dir_pattern)
	else:
		test_dir_pattern = None

	# status prints
	print('{:56} [{data_path}]'.format(
		'Extract annotations from directory:',
		data_path=data_path))
	print('{:>5} {:50} [{mode}]'.format(
		'...',
		'encoding mode:',
		mode="relative (0,1)" if bb_encoding_0_1 else "absolute [0,width/height]"))
	print('{:>5} {:50} [{mode}]'.format(
		'...',
		'split annotations into train/test sets:',
		mode='{status}'.format(
			status=train_test_split,
			)
		if not train_test_split else
		'{status};\t*{directory}*'.format(
			status=train_test_split,
			directory=test_dir_pattern,
			)
			)
		)

	# generate lists containing markup files for train and test sets
	train_test_lists = _get_annotation_files(data_path,test_dir_pattern)

	# extract annotation for each item in the lists
	for i in range(len(train_test_lists)):
		markupFiles = train_test_lists[i]
		key = TRAINING_DATA_KEY
		if i == 1:
			key = TEST_DATA_KEY
		for markupFile in markupFiles:
			item = _extract_from(
				markupFile,
				bb_encoding_0_1,
				**kwargs,
				)
			data[key].append(item)

	return data

def _process_coco_annotations(data_path,bb_encoding_0_1=False,test_data_fraction_n_in_10=3,**kwargs):
	pass
	if not os.path.isfile(data_path):
		raise Exception('provided path `{}` must point to a file.'.format(data_path))
	mod = 11
	if test_data_fraction_n_in_10 > 0:
		mod = 10 // test_data_fraction_n_in_10
	data = dict({
		TRAINING_DATA_KEY:list(),
		TEST_DATA_KEY:list(),
		})
	data_api = coco.COCO(data_path)
	images = data_api.dataset['images']
	annotations = data_api.dataset['annotations']
	annos_for_img_map = {}
	for i in range(len(annotations)):
		anno = annotations[i]
		if anno['image_id'] not in annos_for_img_map:
			annos_for_img_map[anno['image_id']] = []
		annos_for_img_map[anno['image_id']].append(anno['id'])
	i = 0
	for img_id,anno_ids in iter(annos_for_img_map.items()):
		if len(anno_ids) == 0:
			continue
		img = data_api.loadImgs([img_id])
		annos = data_api.loadAnns(anno_ids)
		cats = data_api.loadCats([anno['category_id'] for anno in annos])
		item = _generate_coco_data_dict(img,annos,cats)
		_process_data(item,bb_encoding_0_1)
		j = (i % 10)+1
		key = TRAINING_DATA_KEY
		if j % mod == 0:
			key = TEST_DATA_KEY
		data[key].append(item)
		i += 1
	return data

def process(data_path,**kwargs):
	''' Processes data generation accroding to the provided data_path parameter.

	Args:
		data_path: path to a directory or to a coco annotations json file
		kwargs: dictionary of kwargs that is passed to successive functions

	Returns:
		data: dictionary of bounding box annotations with lists of ground truth
			annotations that are identified by the following keys; a ground
			truth item it dictionary that represents the xml markup structure
			- train: contains the dataset of training items
			- test: contains the dataset of test/validation items
	'''
	if os.path.isdir(data_path):
		data = _process(data_path,**kwargs)
	else:
		data = _process_coco_annotations(data_path,**kwargs)
	# status prints
	train_data = len(data[TRAINING_DATA_KEY])
	test_data = len(data[TEST_DATA_KEY])
	print('{:>5} {:05d} {:44} [{:05d} training / {:05d} testing]'.format(
		'...',
		train_data+test_data,
		'items loaded:',
		train_data,
		test_data))
	return data

def generate_auto_annotated_training_data(root_data_dir,train_data_dir,ckpt_pattern):
	queue = ['']
	while len(queue) > 0:
		subdir = queue.pop()
		subdir_auto_anno = os.path.join(subdir,ckpt_pattern)
		abs_subdir_auto_anno = os.path.abspath(os.path.join(root_data_dir,subdir_auto_anno))
		if os.path.isdir(abs_subdir_auto_anno):
			if not os.path.isdir(os.path.abspath(os.path.join(train_data_dir,subdir))):
				try:
					anno_target = os.path.abspath(os.path.join(train_data_dir,ckpt_pattern,subdir))
					os.makedirs(anno_target)
				except:
					pass
				finally:
					markups = os.listdir(abs_subdir_auto_anno)
					for markup in markups:
						file_path = os.path.abspath(os.path.join(abs_subdir_auto_anno,markup))
						file_target_path = os.path.abspath(os.path.join(anno_target,markup))
						if not os.path.isfile(file_target_path):
							print('copy {} to {}'.format(file_path,anno_target))
							shutil.copy(file_path, anno_target)
		else:
			items = os.listdir(os.path.abspath(os.path.join(root_data_dir,subdir)))
			for item in items:
				item_subdir_path = os.path.join(subdir,item)
				if os.path.isdir(os.path.abspath(os.path.join(root_data_dir,item_subdir_path))):
					queue.append(item_subdir_path)
	pass

def generate_training_data(data_path,training_name,test_dir_pattern,test_frac=0.3,training_root_dir=None,**kwargs):
	''' Recursively searches for bounding box annotation markup files in the data_path
	directory and copies the annotation files into a new directory for the training
	process. The new directory is created either under the training_root_dir directory
	or as a subdirectory in the data_path. The data is split according to the provided
	fraction parameter into train and test dataset.

	Args:
		data_path: root directory of the bounding box annotations
		training_name: name of the training process
		test_dir_pattern: name of the directory into which the test dataset should
			be stored
		test_frac: [default:0.3] fraction of the test data
		training_root_dir: output directory

	Returns:
		new_data_path: root path to the training directory. The training directory
			is created in under the data_path (if no training_root_dir is specified)
			or under the training_root_dir. The datasets for training and validation
			are put into the annotations subdirectory.
	'''
	#annotation_files = _get_annotation_files(data_path,None)
	annotation_files = _get_annotation_files(data_path,test_dir_pattern)

	new_data_path = os.path.join(data_path,training_name,TRAINING_ANNOTATIONS_SUBDIR)
	if training_root_dir is not None:
		new_data_path = os.path.join(training_root_dir,training_name,TRAINING_ANNOTATIONS_SUBDIR)

	split_info = [0,0]
	for markupFiles in annotation_files:
		for markupFile in markupFiles:
			sub_folder = TRAINING_DATA_KEY
			if random.random() < test_frac:
				sub_folder = test_dir_pattern
			markup_path = markupFile[len(data_path):]
			path_fractions = markup_path.split('/')
			target_dir = os.path.join(new_data_path,sub_folder,*path_fractions[:-1])
			try:
				os.makedirs(target_dir)
			except:
				pass
			finally:
				shutil.copy(markupFile, target_dir)
				if sub_folder == TRAINING_DATA_KEY:
					split_info[0]+=1
				else:
					split_info[1]+=1
	print('Generated training data:({} items)\ttest data:({} items)'.format(*split_info))
	return new_data_path

def main(**kwargs):
	'''
	Runs a round of annotation extraction and
	exports the data to a json file if the export flag is provided.
	If the export directory does not exist, it will be created.

	# Args:
		kwargs dictionary of keyword parameters that are passed to subsequent function calls
	'''
	try:
		training_name = kwargs['training_name']
		if training_name is not None:
			kwargs['data_path'] = generate_training_data(**kwargs)

		# root path is defined
		data = process(**kwargs)

		export_flag = kwargs.get('export_flag',False)
		if not export_flag:
			# no export
			sys.exit(0)
		else:
			# ensure export dir exists
			try:
				# create directory and generate folder info file
				os.makedirs(EXPORT_DIR)
				with open(os.path.join(EXPORT_DIR,'info.txt'),'w') as f:
					f.write('This directory is auto-generated and used by the extractAnnotations.py script to store exported annotation data as json file.')
			except:
				# EXPORT_DIR already exists
				pass

			# export part
			train_data = len(data[TRAINING_DATA_KEY])
			test_data = len(data[TEST_DATA_KEY])
			if (train_data > 0 or test_data > 0):
				export_path = os.path.join(EXPORT_DIR,kwargs.get('export_file_name'))
				prefix,extension = os.path.splitext(export_path)
				i = 1
				while True:
					if os.path.exists(export_path):
						# change name of export file
						export_path = '{}({:d}){}'.format(
							prefix,
							i,
							extension)
						i += 1
					else:
						print('annotations are exported to: {}'.format(export_path))
						break
				# write file to EXOIRT_DIR
				with open(export_path,'w') as f:
					f.write(json.dumps(data))

	except EnvironmentError as e:
		print('environment not set properly: {}\nplease provide a root directory for extracting annotations via the -i or --imageroot parameter'.format(e))
	except Exception as e:
		print(e)
		print('extractAnnotations.py -i <imageRootDirectory> -e <exportFileName>')

	sys.exit(0)

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

	# call main
	main(**kwargs)
