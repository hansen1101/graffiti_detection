'''
Automatically generate annotations for a set of images 
using a tensorflow object detection api graph.

# Execution examples (call from Graffiti_Detection/ directory):
python generateAnnotationFromModelsPredictions.py \
	--path_to_inference_graph=models/tensorflow_models/ssd_inception_v2_graffiti/ssd_inception_v2_training_2/export/ckpt-64000/frozen_inference_graph.pb \
	--path_to_images=/usr/local/share/data/graffiti_images/flickr \
	--path_to_label_map=data/generateTfRecord_test_training/label_map.pbtxt \
	--confidence_threshold=0.35 \
	--separate_annotations

python generateAnnotationFromModelsPredictions.py \
	--frozen_model_name=ssd_inception_v2_training_2 \
	--checkpoint_num=64000 \
	--path_to_images=/usr/local/share/data/graffiti_images/flickr \
	--path_to_label_map=data/generateTfRecord_test_training/label_map.pbtxt \
	--confidence_threshold=0.35 \
	--separate_annotations

python generateAnnotationFromModelsPredictions.py \
	--model_root_path=models/tensorflow_models/ssd_inception_v2_training_2 \
	--checkpoint_num=64000 \
	--path_to_images=/usr/local/share/data/graffiti_images/flickr \
	--path_to_label_map=data/generateTfRecord_test_training/label_map.pbtxt \
	--confidence_threshold=0.35 \
	--separate_annotations

# Args:
	-g --path_to_inference_graph path to tensorflow object detection api inference graph that should be used for prediction
	-i --path_to_images directory from where to start searching for images to predict and derive the annotation for
	-l --path_to_label_map label map for which the tf records should be generated
	-t --confidence_threshold float between 0.0 and 1.0 indicating the min confidence a bounding box should have to get included into the annotation file
	-c --checkpoint_num number of the checkpoint to be usee
	-f --frozen_model_name name of the model that should be used for inference (substitution for path to inference graph in combination with checkpoint num)
	-m --model_root_path path to the model's root directory
	
	-s --separate_annotations flag only parameter, if set the annotations files are written to a separate subdir in the images folder
'''

import sys
import os
import getopt
import re
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from lxml import etree
from utils import console_utils
from object_detection.utils import label_map_util

PARAMS = [
	[
		("g","path_to_inference_graph",),
		("i","path_to_images",),
		("l","path_to_label_map",),
		("t","confidence_threshold",),
		("c","checkpoint_num",),
		("f","frozen_model_name",),
		("m","model_root_path",),
		],
	[
		('s','separate_annotations',),
		],
]

DEFAULT_LABEL_MAP = {'graffiti':1,'graffiti_area':2}

DATABASE = 'Unknown'
SEGMENTED = 0
POSE = 'Unspecified'
TRUNCATED = 0
DIFFICULT = 0
CONFIDENCE_THRESHOLD = 0.3
MODELS_SUBDIR = ['models','tensorflow_models',None,'export',None,'frozen_inference_graph.pb']
MARKUP_FILE_KEY = 'ckpt-'
FROZEN_INFERENCE_GRAPH_FILE_NAME = 'frozen_inference_graph.pb'

EXAMPLE_GRAPH = 'models/tensorflow_models/ssd_inception_v2_training_2/ckpt-64000/frozen_inference_graph.rb'

def _checkpoint_folder_id(checkpoint_num='0'):
	'''
	'''
	return MARKUP_FILE_KEY+checkpoint_num

def _path_to_tf_inference_graph(frozen_model,checkpoint_num,entry_index=0):
	'''Generates a path to a frozen inference graph.
	'''
	path = MODELS_SUBDIR
	path[2] = frozen_model
	path[4] = _checkpoint_folder_id(checkpoint_num)
	return os.path.join(*path[entry_index:])

def _checkpoint_num(path):
	'''
	Extracts a checkpoint number from a path 
	(e.g. path to a frozen inference graph).

	# Args:
		path: string holding a ckpt flag

	# Return:
		number of chekpoint
	'''
	ckpt_pattern = re.compile('^.*/ckpt-(\d+)/.*$')
	m = re.match(ckpt_pattern,path)
	if m:
		return m.groups()[0]
	return '0'

def load_tf_graph(path_to_tf_inference_graph):
	'''
	Loads and returns a tensorflow object detection api graph
	'''
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(path_to_tf_inference_graph, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')
	return detection_graph

def run_inference_for_single_image(image,graph):
	'''Runs the forward pass of a single image 
	through the graph, creates and returns the output mapping.

	# Args:
		image: numpy array encoding of an image
		graph: tensorflow graph

	# Return:
		output_dict: a dictionary containing the following fields
		- num_detections: int value
		- detection_classes: [num_detections]
		- detection_boxes: [num_detections, 4]
		- detection_scores: [num_detections]
		- detection_masks: [num_detections, mask_height, mask_width] (optional)
	'''
	with graph.as_default():
		with tf.Session() as sess:
			# Get handles to input and output tensors
			ops = tf.get_default_graph().get_operations()
			all_tensor_names = {output.name for op in ops for output in op.outputs}
			tensor_dict = {}
			for key in [
				'num_detections', 'detection_boxes', 'detection_scores',
				'detection_classes', 'detection_masks'
			]:
				tensor_name = key + ':0'
				if tensor_name in all_tensor_names:
					tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
			
			if 'detection_masks' in tensor_dict:
				
				# The following processing is only for single image
				detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
				detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
				
				# Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
				real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
				detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
				detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
				detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
				detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
			
				# Follow the convention by adding back the batch dimension
				tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
			
			image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

			# Run inference
			output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})

			# all outputs are float32 numpy arrays, so convert types as appropriate
			output_dict['num_detections'] = int(output_dict['num_detections'][0])
			output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
			output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
			output_dict['detection_scores'] = output_dict['detection_scores'][0]
			if 'detection_masks' in output_dict:
				output_dict['detection_masks'] = output_dict['detection_masks'][0]
	return output_dict

def generate_markup_file_from_detection_dict(image_path,label_map_dict,detection_graph,cofidence_threshold,markup_file_folder_name,**kwargs):
	'''Generates xml markup file from model predictions
	for given image.

	Generates an xml tree and adds in the bounding boxes 
	generated by the predictions for the image.
	Bounding boxes are added if the corresponding confidence 
	is higher than the threshold.
	
	Saves the markup file to disk either in the image's root 
	directory or into a subdirectory (name provided by 
	parameter) of the image's root directory.
	
	# Args:
		image_path full path to the image
		label_map_dict label map of the model that maps 
			indices to class names
		detection_graph path to graph of frozen tensorflow 
			object detection api model
		cofidence_threshold threshold for positive/negative 
			detection decisions
		markup_file_folder_name directory name where the 
			markup file should be stored
	'''	
	
	markup_file_path,_ = os.path.splitext(image_path)
	path_fractions = markup_file_path.split('/')
	file_folder = os.path.join('/',*path_fractions[:len(path_fractions)-1],markup_file_folder_name)
	file_name = path_fractions[len(path_fractions)-1]
	file_name += '.xml'
	file_path = os.path.join(file_folder,file_name)
	
	# generate xml tag values
	path_fractions = image_path.split('/')
	image_folder = path_fractions[len(path_fractions)-2]
	image_file_name = path_fractions[len(path_fractions)-1]
	try:
		image_np = img_to_array(load_img(image_path))
		root = etree.Element('annotation')
		
		folder = etree.Element('folder')
		folder.text = image_folder
		root.append(folder)
		
		filename = etree.Element('filename')
		filename.text = image_file_name
		root.append(filename)
		
		path = etree.Element('path')
		path.text = image_path
		root.append(path)

		source = etree.Element('source')
		database = etree.Element('database')
		database.text = DATABASE
		source.append(database)
		root.append(source)

		# the array based representation of the image will be used later in order to prepare the
		# result image with boxes and labels on it.
		img_height,img_width,img_channels = image_np.shape

		size_node = etree.Element('size')
		width_node = etree.Element('width')
		width_node.text = '{}'.format(img_width)
		size_node.append(width_node)
		height_node = etree.Element('height')
		height_node.text = '{}'.format(img_height)
		size_node.append(height_node)
		dim_node = etree.Element('depth')
		dim_node.text = '{}'.format(img_channels)
		size_node.append(dim_node)
		root.append(size_node)
		root.append(source)
		

		segmented_node = etree.Element('segmented')
		segmented_node.text = '{}'.format(SEGMENTED)
		root.append(segmented_node)

		# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		image_np_expanded = np.expand_dims(image_np, axis=0)

		# Actual detection.
		output_dict = run_inference_for_single_image(image_np, detection_graph)
		boxes = output_dict.get('detection_boxes',[])
		classes = output_dict.get('detection_classes',[])
		scores = output_dict.get('detection_scores',[])

		for i,score in enumerate(scores):
			if score >= cofidence_threshold:
				
				object_node = etree.Element('object')

				box = boxes[i]
				object_class = 'N/A'
				for name,index in label_map_dict.items():
					if index == classes[i]:
						object_class = name

				name_node = etree.Element('name')
				name_node.text = object_class
				object_node.append(name_node)

				pose_node = etree.Element('pose')
				pose_node.text = POSE
				object_node.append(pose_node)

				truncated_node = etree.Element('truncated')
				truncated_node.text = '{}'.format(TRUNCATED)
				object_node.append(truncated_node)

				difficult_node = etree.Element('difficult')
				difficult_node.text = '{}'.format(DIFFICULT)
				object_node.append(difficult_node)

				box_matrix = np.reshape(box,(2,2,))
				size_vector = np.array([img_height,img_width])
				normalized_matrix = box_matrix * size_vector
				normalized_box = np.reshape(normalized_matrix,(-1))
				ymin, xmin, ymax, xmax = np.round_(normalized_box,decimals=0).astype('int')
				
				bndbox_node = etree.Element('bndbox')
				xmin_node = etree.Element('xmin')
				xmin_node.text = '{}'.format(xmin)
				bndbox_node.append(xmin_node)
				ymin_node = etree.Element('ymin')
				ymin_node.text = '{}'.format(ymin)
				bndbox_node.append(ymin_node)
				xmax_node = etree.Element('xmax')
				xmax_node.text = '{}'.format(xmax)
				bndbox_node.append(xmax_node)
				ymax_node = etree.Element('ymax')
				ymax_node.text = '{}'.format(ymax)
				bndbox_node.append(ymax_node)
				object_node.append(bndbox_node)
				root.append(object_node)

		tree = etree.ElementTree(root)
		try:
			os.makedirs(file_folder)
		except:
			pass
		tree.write(file_path,pretty_print=True,encoding='utf-8')
		print('{:<35}{}'.format('successfully processed item:',image_file_name))
	except Exception as e:
		#print(e)
		print('{:<35}{}'.format('skipped processing of item:',image_file_name))

def generate_kwargs_dict(opts,args):
	'''
	Generates parameter dictionary for internal use from provided opts and args typles.
	Checks presence for each short/long option pair provided by PARAMS in the opts list
	and extracts the provided value	if option is found.

	# Args:
		opts: list of of ([long]name, value) pairs holding, names could be included in PARAMS
		args: list of additional args provided through execution call

	# Returns:
		kwargs: dictionary containing the following fields
			path_to_tf_inference_graph: path to file
			checkpoint_num: number of a checkpoint
			image_dir_root_path: directory containing images
			path_to_label_map: path to label map for the model
			cofidence_threshold: score between 0 and 1
			markup_file_folder_name: name of subdir for annotation files
	'''
	kwargs = {}
	
	checkpoint_num = console_utils.extract_option(PARAMS[0][4],opts)
	path_to_tf_inference_graph = console_utils.extract_option(PARAMS[0][0],opts)
	if path_to_tf_inference_graph is None:
		if checkpoint_num is None:
			raise EnvironmentError('parameter required --checkpoint_num / -n (e.g. --checkpoint_num=71113)')

		frozen_model = console_utils.extract_option(PARAMS[0][5],opts)
		if frozen_model is None:
			frozen_model = console_utils.extract_option(PARAMS[0][6],opts)
			if frozen_model is None:
				raise EnvironmentError('provide path to model: --model_root_path / -m (e.g. --model_root_path=models/tensorflow_models/ssd_inception_v2_graffiti)\nor provide model name: --frozen_model_name / -f (e.g. --frozen_model_name=ssd_inception_v2_graffiti)')
			path_to_tf_inference_graph = _path_to_tf_inference_graph(frozen_model,checkpoint_num,entry_index=2)
		else:
			path_to_tf_inference_graph = _path_to_tf_inference_graph(frozen_model,checkpoint_num)
	else:
		if os.path.isdir(path_to_tf_inference_graph):
			path_to_tf_inference_graph = os.path.join(path_to_tf_inference_graph,FROZEN_INFERENCE_GRAPH_FILE_NAME)
	
		if checkpoint_num is None:
			checkpoint_num = _checkpoint_num(path_to_tf_inference_graph)

	if not os.path.isfile(path_to_tf_inference_graph):
		raise EnvironmentError('no valid tensorflow graph provided: --path_to_inference_graph / -g (e.g. --path_to_inference_graph={})'.format(EXAMPLE_GRAPH))

	kwargs['path_to_tf_inference_graph'] = path_to_tf_inference_graph
	kwargs['checkpoint_num'] = checkpoint_num
	#kwargs['checkpoint_folder_id'] = _checkpoint_folder_id(checkpoint_num)
		

	image_dir_root_path = console_utils.extract_option(PARAMS[0][1],opts)
	if image_dir_root_path is None:
		raise EnvironmentError('provide path to image directory: --path_to_images / -i (e.g. --path_to_images=/usr/local/share/data/graffiti_images/easydb)')
	else:
		kwargs['image_dir_root_path'] = image_dir_root_path

	kwargs['path_to_label_map'] = console_utils.extract_option(PARAMS[0][2],opts)

	cofidence_threshold = console_utils.extract_option(PARAMS[0][3],opts)
	if cofidence_threshold is None:
		cofidence_threshold = CONFIDENCE_THRESHOLD
	kwargs['cofidence_threshold'] = float(cofidence_threshold)
	
	kwargs['markup_file_folder_name'] = ''
	write_annotations_into_separate_directory = console_utils.check_flag_presence(PARAMS[1][0],opts)
	if write_annotations_into_separate_directory:
		kwargs['markup_file_folder_name'] += _checkpoint_folder_id(checkpoint_num)
		kwargs['markup_file_folder_name'] += '_threshold={:.5}'.format(cofidence_threshold)
	
	return kwargs

def main(**kwargs):
	'''
	Recursively turns model predictions into annotation files
	'''
	detection_graph = load_tf_graph(kwargs['path_to_tf_inference_graph'])
	try:
		label_map_dict = label_map_util.get_label_map_dict(kwargs['path_to_label_map'])
	except:
		label_map_dict = DEFAULT_LABEL_MAP
	
	image_dir_root_path = kwargs['image_dir_root_path']
	dir_queue = [os.path.abspath(image_dir_root_path)]
	while len(dir_queue) > 0:
		directory = dir_queue.pop(0)
		print(directory)
		items = [os.path.join(directory,i) for i in os.listdir(directory)]
		for item in items:
			if os.path.isdir(item):
				dir_queue.append(item)
			else:
				generate_markup_file_from_detection_dict(item,label_map_dict,detection_graph,**kwargs)

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