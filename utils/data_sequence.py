import os
import math
import keras
import numpy as np
from keras.utils.data_utils import Sequence
from keras.preprocessing.image import load_img, img_to_array
from utils.classification_auxiliary import tf_preprocessor, invert_tf_preprocessor
from utils import similarity, extractAnnotations
from PIL import Image
import urllib
import requests
from io import BytesIO
import time
#from

DEFAULT_SIMILARITY_FUNCTION = 'jaccard_similarity'
GT_IOU_DROP_UNDER = -1.0
MAX_ANCHOR_LIMIT = -1.0

class Keras_Detection_Sequence(Sequence):
	''' Implementation of Keras sequence of data for object detection.
	This class provides the required methods to manage default anchor boxes
	for object detection.
	Especially methods for anchor box encodings and translation between
	offset values (predicitons) and regular bounding box encodings (ground truth)
	are provided.
	'''
	ASPECT_RATIOS = [1.,2.,3.,1/2,1/3]

	@staticmethod
	def convert_norm_to_min_max(box):
		ymin = box[0]-(box[2]/2.)
		ymax = box[0]+(box[2]/2.)
		xmin = box[1]-(box[3]/2.)
		xmax = box[1]+(box[3]/2.)
		return np.array([ymin,xmin,ymax,xmax])

	@staticmethod
	def convert_min_max_to_norm(box):
		h = box[2]-box[0]
		cy = box[0]+h/2.
		w = box[3]-box[1]
		cx = box[1]+w/2.
		return np.array([cy,cx,h,w])

	@staticmethod
	def clip_to_image(box_array):
		min_max_box = Keras_Detection_Sequence.convert_norm_to_min_max(box_array)
		box = np.where(
			np.logical_or(
				min_max_box < 0.,
				min_max_box > 1.),
			np.array([0.,0.,1.,1.]),
			min_max_box)
		return Keras_Detection_Sequence.convert_min_max_to_norm(box)

	@staticmethod
	def anchor_is_in_img_bounds(cy,cx,h,w):
		''' Checks if box encoding crosses image boundaries.

		# Args: relative bounding box encodings with
			cy: center y coordinate
			cx: center x coordinate
			h: height of the box
			w: width of the box

		# Returns:
			true if anchor box does not cross image boundaries,
			false otherwise
		'''
		h1 = 0. <= cy + h/2. <= 1.
		h2 = 0. <= cy - h/2. <= 1.
		w1 = 0. <= cx + w/2. <= 1.
		w2 = 0. <= cx - w/2. <= 1.
		return h1 and h2 and w1 and w2

	@staticmethod
	def _anchors_per_grid_cell(aspect_ratios):
		''' Calculates the number of default anchor boxes per feature map location.
		'''
		if 1. in aspect_ratios:
			return len(aspect_ratios)+1
		return len(aspect_ratios)

	@staticmethod
	def _compute_scale_for_map(k,num_feat_maps,min_scale,max_scale):
		''' Computes scale for level k in the feature pyramid
		with num_feat_maps levels with respect to min and max scale
		values.

		# Returns:
			s_k: scale for feature map at level k
		'''
		return min_scale + (max_scale - min_scale) / (num_feat_maps - 1) * k

	@staticmethod
	def _compute_anchor_dimensions(scale,aspect_ratio,max_anchor_limit=MAX_ANCHOR_LIMIT,**kwargs):
		'''
		Computes width and height of default anchor boxes with respect
		to a maximum dimension constraint.

		# Returns:
			tuple with values for
				new_height: min(max_anchor_limit,height)
				new_width: min(max_anchor_limit,width)
		'''
		height = scale / math.sqrt(aspect_ratio)
		width = scale * math.sqrt(aspect_ratio)
		if max_anchor_limit > 0.:
			height = min(max_anchor_limit,height)
			width = min(max_anchor_limit,width)
		return height,width

	@staticmethod
	def _compute_anchor_center(i,j,grid_size):
		return (i+.5)/grid_size,(j+.5)/grid_size

	@staticmethod
	def generate_anchor_list(grid_sizes,aspect_ratios,min_scale,max_scale,**kwargs):
		''' Generates the full list of default anchor boxes for the feature
		pyramid as a linear list.

		# Args:
			grid_sizes: [num_feat_maps] resolutions of the feature maps
			aspect_ratios: list of aspect ratios that should be applied
				to each feature map location
			min_scale: lower scaling boundary
			max_scale: upper scaling boundary
			max_anchor_limit:

		# Returns:
			anchor_list: [num_default_anchor_boxes,4] numpy array of
				default anchor box encodings
		'''
		num_feat_maps = len(grid_sizes) # number of different feature maps
		assert num_feat_maps > 1, ('number of feature maps must at least be 2 or greater')
		anchor_list = []
		for k in range(num_feat_maps):
			s_k = Keras_Detection_Sequence._compute_scale_for_map(k,num_feat_maps,min_scale,max_scale)
			grid_size = grid_sizes[k]
			for i in range(grid_size):
				for j in range(grid_size):
					cy,cx = Keras_Detection_Sequence._compute_anchor_center(i,j,grid_size)
					for a in aspect_ratios:
						h,w = Keras_Detection_Sequence._compute_anchor_dimensions(s_k,a,**kwargs)
						anchor_list.append([cy,cx,h,w])
						if a == 1.:
							s_k_1 = Keras_Detection_Sequence._compute_scale_for_map(k+1,num_feat_maps,min_scale,max_scale)
							h,w = Keras_Detection_Sequence._compute_anchor_dimensions(math.sqrt(s_k*s_k_1),a,**kwargs)
							anchor_list.append([cy,cx,h,w])
		return np.array(anchor_list)

	@staticmethod
	def _fit_to_model_outputs(linear_anchor_array,grid_sizes,num_anchors_per_location):
		''' Converts a linear anchor list into anchor matrix according to provided grid sizes.
		'''
		index = 0
		split_indices = []
		for size in grid_sizes[:-1]:
			index += size*size*num_anchors_per_location
			split_indices.append(index)
		linear_feature_map_anchors = np.split(linear_anchor_array,split_indices,axis=0)

		assert len(linear_feature_map_anchors) == len(grid_sizes), 'not a split for each grid size'
		feature_map_anchor_list = []
		for i,size in enumerate(grid_sizes):
			feature_map_anchors = linear_feature_map_anchors[i]
			feature_map_anchor_list.append(
				np.reshape(
					feature_map_anchors,
					(size,size,num_anchors_per_location,-1),
					)
				)

		'''
		@TODO assert correct structur
		print(split_indices)
		print(len(linear_anchor_target_array))

		print(feature_map_anchor_list[0][0,0,0,:])
		print(feature_map_anchor_list[1][0,0,0,:])
		print(feature_map_anchor_list[2][0,0,0,:])
		print(feature_map_anchor_list[3][0,0,0,:])

		print(feature_map_anchor_list[0].shape)
		print(feature_map_anchor_list[1].shape)
		print(feature_map_anchor_list[2].shape)
		print(feature_map_anchor_list[3].shape)
		'''
		return feature_map_anchor_list

	@staticmethod
	def _invert_fit_to_model_outputs(feature_map_anchor_list,grid_sizes,num_anchors_per_location):
		'''
		Converts nested lists of prediction encodings into a linear list
		that corresponds to the linear anchor box array.

		Args:
			feature_map_anchor_list: nested list [num_feat_maps] where
				each list at feature map level i has shape
				[grid_size[i],grid_size[i],num_anchors,5]
			grid_sizes: shape of the feature pyramid
			num_anchors_per_location: number of predictions per location

		Return:
			linear_anchor_vector: numpy array of shape (num_anchors,5),
			  array contains the predictions in a linear arrangement that
			  corresponds to the default anchor list self.linear_anchor_array
			  and has the same shape.

			list_of_linear_anchor_vectors_per_grid: list of numpy arrays of
			  shape (H_i*W_i*anchors_per_location,5), where H_i is the grid size
			  of feature map i.
			  The listing of linear predictions corresponds to the feature map
			  structure.
		'''
		linear_anchor_vector = None
		list_of_linear_anchor_vectors_per_grid = []
		anchor_number = np.sum(np.array([size*size*num_anchors_per_location for size in grid_sizes]))
		for i,size in enumerate(grid_sizes):
			feature_map = feature_map_anchor_list[i]
			num_anchors_in_map = np.prod(np.array(feature_map.shape[:-1]))
			linear_anchor_array = np.reshape(feature_map,(num_anchors_in_map,-1))

			# append to list and concat to vector
			list_of_linear_anchor_vectors_per_grid.append(linear_anchor_array)
			if linear_anchor_vector is None:
				linear_anchor_vector = linear_anchor_array
			else:
				linear_anchor_vector = np.concatenate((linear_anchor_vector,linear_anchor_array), axis=0)
			anchor_number -= num_anchors_in_map
		assert anchor_number == 0, (
				'feature map anchor shape '
				'does not fit to grid size')
		assert np.array_equal(linear_anchor_vector.shape,(np.sum(np.array(grid_sizes)*np.array(grid_sizes))*num_anchors_per_location,5)), (
				'predictions and default anchor '
				'list should have the same shape')
		return linear_anchor_vector,list_of_linear_anchor_vectors_per_grid

	@staticmethod
	def generate_regression_targets(anchor_box,target_box):
		ty = (target_box[0]-anchor_box[0])/anchor_box[2]
		tx = (target_box[1]-anchor_box[1])/anchor_box[3]
		th = math.log(target_box[2]/anchor_box[2])
		tw = math.log(target_box[3]/anchor_box[3])
		return np.array([ty,tx,th,tw])

	@staticmethod
	def invert_generate_regression_targets(anchor_box,prediction_box,clip_to_image):
		''' Translates bounding box offset values into relative
		bounding box encodings.

		# Args:
			anchor_box: [4] default anchor box that corresponds to prediction
			prediction_box: [5] prediction vector
			clip_to_image: boolean if true, final box encoding
				is clipped to the image boundaries

		# Returns:
			box: [4] relative bounding box encodings
		'''
		ty = prediction_box[0]*anchor_box[2]+anchor_box[0]
		tx = prediction_box[1]*anchor_box[3]+anchor_box[1]
		try:
			th = math.exp(prediction_box[2])*anchor_box[2]
		except:
			print(anchor_box[2])
			th = 1.
		try:
			tw = math.exp(prediction_box[3])*anchor_box[3]
		except:
			print(anchor_box[3])
			tw = 1.
		box = np.array([ty,tx,th,tw])
		if clip_to_image and not Keras_Detection_Sequence.anchor_is_in_img_bounds(*box):
			box = Keras_Detection_Sequence.clip_to_image(box)
		return box

	@staticmethod
	def read_img_into_np_array(img_path,height,width,tf_transformation=True,verbose=False):
		''' Open image from either path on disk or url and
		transforms it into numpy array of given size.
		Data can be loaded with (required by keras models)
		or without (required by tensorflow object detection
		api graphs) preprocessing.

		Args:
			img_path: path to image on disk or url
			height: int
			width: int
			tf_transformation: bool (default:true) apply data
				transformation that is required bu keras models
			verbose: bool (default:false) output errors if true

		Returns:
			img_data: [height,width,3] preprocessed image data

		'''
		img_data = None
		try:
			if urllib.parse.urlparse(img_path).scheme != "":
				print('path is a url: {}'.format(img_path))
				response = requests.get(img_path)
				if response.status_code == 200:
					img = Image.open(BytesIO(response.content))
			else:
				img = load_img(
					img_path,
					color_mode='rgb',
					target_size=(height,width))
			img_data = img_to_array(img).astype(int)
			if tf_transformation:
				img_data = tf_preprocessor(img_data)
		except Exception as e:
			if verbose:
				print(e)
				#print('could not create data for file: {}'.format(img_path))
		finally:
			return img_data

	def fit_to_model_outputs(self,linear_anchor_array):
		return self._fit_to_model_outputs(linear_anchor_array,self.grid_sizes,self.num_anchors_per_location)

	def invert_fit_to_model_outputs(self,anchor_map):
		''' Transforms an anchor map into a linear list of anchors.
		# Returns:
			@see _invert_fit_to_model_outputs
		'''
		linear_anchor_vector,list_of_linear_anchor_vectors_per_grid = self._invert_fit_to_model_outputs(anchor_map,self.grid_sizes,self.num_anchors_per_location)
		return linear_anchor_vector,list_of_linear_anchor_vectors_per_grid

	@staticmethod
	def _decode_predictions(prediction_box_list,anchor_box_list,downrate_out_off_bounds,clip_to_image):
		''' Translates predicted offsets values into proper bounding
		box encodings according to corresponding default anchor boxes.

		Args:
			prediction_box_list: [num_anchors,5]
			anchor_box_list: [num_anchors,4]
			clip_to_image: bool predictions are clipped to image
				boundaries if value is true
			downrate_out_off_bounds: bool confidence scores are
				set to zero for all images that cross image
				boundaries if true; if clip_to_image
				is true, this parameter has no effect.

		Returns:
			result_list: [num_anchors,5] numpy array of
				converted predictions
		'''
		def sigmoid(x):
			conf = 0.
			try:
				conf = 1.0 / (1.0 + np.exp(-x))
			except:
				if x > 0:
					conf = 1.
			return conf
		assert prediction_box_list.shape[:1] == anchor_box_list.shape[:1], (
			'shape of predictions ',
			prediction_box_list.shape,
			' is not compatible with shape of anchor box list ',
			anchor_box_list.shape)
		result_list = None
		for i in range(prediction_box_list.shape[0]):
			bounding_box_pred = Keras_Detection_Sequence.invert_generate_regression_targets(anchor_box_list[i],prediction_box_list[i],clip_to_image)
			if Keras_Detection_Sequence.anchor_is_in_img_bounds(*bounding_box_pred) or not downrate_out_off_bounds:
				confidence = sigmoid(prediction_box_list[i][4])
			else:
				confidence = 0.
			bounding_box_pred = np.append(bounding_box_pred,confidence)
			bounding_box_pred = np.expand_dims(bounding_box_pred,axis=0)
			# bounding_box_pred.shape = [1,5]
			if result_list is None:
				result_list = np.copy(bounding_box_pred)
			else:
				result_list = np.append(result_list,bounding_box_pred,axis=0)
		assert result_list.shape == prediction_box_list.shape, (
			'decoded result list should have '
			'same shape as predicted offsets list')
		return result_list

	def decode_predictions(self,prediction_box_list,downrate_out_off_bounds=False,clip_to_image=True,**kwargs):
		result_list = self._decode_predictions(prediction_box_list,self.linear_anchor_array,downrate_out_off_bounds,clip_to_image)
		return result_list

	def decode_predictions_sorted(self,prediction_box_list,downrate_out_off_bounds=False,clip_to_image=True,**kwargs):
		result_list = self._decode_predictions(prediction_box_list,self.linear_anchor_array,downrate_out_off_bounds,clip_to_image)
		sorted_result_list = result_list[(-1*result_list)[:,4].argsort()]
		return sorted_result_list

	def decode_map_list_predictions(self,linear_map_list,downrate_out_off_bounds=False,clip_to_image=True,**kwargs):
		''' Returns a list of sorted linear decoded predictions
		shape [M,H_m*W_m*A,5]
			M: number of feature maps
			H_m: heights of feature map
			W_m: width of feature map
			A: num anchors per feature map
		'''
		lengths = [npary.shape[0] for npary in linear_map_list]
		tmp_list = None
		for listing in linear_map_list:
			if tmp_list is None:
				tmp_list = np.copy(listing)
			else:
				tmp_list = np.append(tmp_list,listing,axis=0)
		result_list = self._decode_predictions(tmp_list,self.linear_anchor_array,downrate_out_off_bounds,clip_to_image)
		i = 0
		sorted_map_list = []
		for length in lengths:
			result_slice = np.copy(result_list[i:i+length,:])
			i += length
			sorted_result_slice = result_slice[(-1*result_slice)[:,4].argsort()]
			sorted_map_list.append(sorted_result_slice)
		return sorted_map_list

	def __init__(self,image_size,grid_sizes,batch_size=4,aspect_ratios=[],min_scale=0.2,max_scale=0.9,preprocess_on_init=False,keras_preprocessing_mode=True,**kwargs):
		'''
		# Args:
			image_size: dimension of the input
			grid_sizes: [num_feat_maps] dimensions of the feature maps
				at different feature pyramid levels
			batch_size: size of the output data chunks
			aspect_ratios: [] set of aspect ratios to be considered
				for the default anchor boxes
			mix_scale: lower scale boundary for an anchor box
			max_scale: upper scale boundary for an anchor box
			preprocess_on_init: boolean indicating whether loading the image
				into a numpy array and the assignment of default anchor boxes
				should be computed once when the image is loaded for the
				first time (memory intense but fast) or everytime the image
				is requested by the model (slow)
			keras_preprocessing_mode: bool (default:True) whether to apply
				additional image transformation (required by all keras models)
				when reading image into numpy array or not.
		'''
		super(Keras_Detection_Sequence,self).__init__()
		print('max anchor limit: {}'.format(kwargs.get('max_anchor_limit',MAX_ANCHOR_LIMIT)))
		print('drop ground truth boxes with lower IoU overlap to any anchor than: {}'.format(kwargs.get('drop_groundtruth_for_iou_lower_than',GT_IOU_DROP_UNDER)))
		self.batch_size = batch_size
		if isinstance(image_size,int):
			self.image_size = (image_size,image_size)
		else:
			self.image_size = image_size
		self.preprocess_on_init = preprocess_on_init
		self.data_transformation = keras_preprocessing_mode

		if preprocess_on_init:
			print('read image at init time')
		else:
			print('read image at batch load time')

		self.grid_sizes = grid_sizes
		if aspect_ratios:
			print('generate anchor encodings for provided aspect_ratios:\t',aspect_ratios)
			self.aspect_ratios = aspect_ratios
		else:
			print('no aspect ratios provided, generate default encodings:\t',self.ASPECT_RATIOS)
			self.aspect_ratios = self.ASPECT_RATIOS
		self.min_scale = min_scale
		self.max_scale = max_scale
		self.num_anchors_per_location = self._anchors_per_grid_cell(self.aspect_ratios)
		self.linear_anchor_array = self.generate_anchor_list(
			self.grid_sizes,
			self.aspect_ratios,
			self.min_scale,
			self.max_scale,
			**kwargs,
			)

	def __getitem__(self, index):
		"""Gets batch at position `index`.
		# Arguments
			index: position of the batch in the Sequence.
		# Returns
			A batch
		"""
		batch_x = self.x[index * self.batch_size:(index+1) * self.batch_size]
		batch_y = self.y[index * self.batch_size:(index+1) * self.batch_size]
		return np.array(batch_x),batch_y

	def __len__(self):
		return int(np.ceil(len(self.x) / float(self.batch_size)))

class Prediction_Data_Sequence(Keras_Detection_Sequence):
	''' Data Sequence generates batches of image arrays and realtive
	paths where	the predictions should be stored.
	'''
	def __init__(self,image_path,max_data_size=-1,**kwargs):
		super(Prediction_Data_Sequence,self).__init__(**kwargs)
		self.x = []
		self.y = []
		if os.path.isdir(os.path.abspath(image_path)):
			root_dir = os.path.abspath(image_path)
			root_dir_parts = root_dir.split('/')
			rel_root_dir = '{}_{:d}'.format(root_dir_parts[-1],int(time.time()))
			dir_queue = [root_dir]
			break_while = False
			while len(dir_queue) > 0 and not break_while:
				directory = dir_queue.pop(0)
				#print('Process items in dir:\t{}'.format(directory))
				img_paths = [os.path.join(directory,i) for i in os.listdir(directory)]
				# read all images and append to x
				for img_path in img_paths:
					if len(self.x)>=max_data_size>=0:
						break_while = True
						break
					if os.path.isdir(img_path):
						dir_queue.append(img_path)
					else:
						img_path_parts = img_path[len(root_dir):].split('/')
						img_sub_dirs = img_path_parts[:-1]

						prefix,file_extension = os.path.splitext(img_path)
						if file_extension == '.xml':
							data = extractAnnotations._extract_from(
								img_path,
								bb_encoding_0_1=True,
								)
							img_path = data['annotation']['path']

						img_file_name = img_path.split('/')[-1]
						img_file_name_prefix,img_file_name_ext = os.path.splitext(img_file_name)

						x = img_path
						y = [os.path.join(rel_root_dir,*img_sub_dirs),img_file_name_prefix,img_file_name_ext]
						img_data = super(Prediction_Data_Sequence,self).read_img_into_np_array(img_path,height=self.image_size[0],width=self.image_size[1],tf_transformation=self.data_transformation)
						if img_data is None:
							continue
						if self.preprocess_on_init:
							x = img_data
							print('loaded item into memory:\t{}'.format(img_path))
						self.x.append(x)
						self.y.append(y)
		else:
			print('{} is not a valid directory'.format(image_path))

	def __getitem__(self, index):
		batch_y = self.y[index * self.batch_size:(index+1) * self.batch_size]
		if self.preprocess_on_init:
			batch_x = self.x[index * self.batch_size:(index+1) * self.batch_size]
		else:
			batch_x = []
			for img_path in self.x[index * self.batch_size:(index+1) * self.batch_size]:
				img_data = super(Prediction_Data_Sequence,self).read_img_into_np_array(img_path,height=self.image_size[0],width=self.image_size[1],tf_transformation=self.data_transformation)
				if img_data is not None:
					batch_x.append(img_data)
		return np.array(batch_x),np.array(batch_y)

class Simple_Bounding_Box_Sequence(Keras_Detection_Sequence):
	''' Data Sequence generates batches of image arrays and ground
	thruth bounding box encodings.
	'''
	def __init__(self,markup_path,**kwargs):
		super(Simple_Bounding_Box_Sequence,self).__init__(**kwargs)
		self.x = []
		self.y = []
		data = extractAnnotations.process(
			markup_path,
			bb_encoding_0_1=True,
		)
		if len(data['test']) > 0:
			print('Warning:\tdata have been splitted, but {} test data are irgnored in further process'.format(len(data['test'])))

		for item_num,item in enumerate(data['train']):
			if (item_num+1) % 500 == 0:
					print('processing item {}/{} ...'.format(item_num+1,len(data['train'])))
			img_path,ground_truth_boxes = Detection_Data_Sequence.unpack_item(item,{})
			if not ground_truth_boxes:
				continue
			x = img_path
			y = np.array(ground_truth_boxes) # shape (num_boxes,4)
			img_data = super(Simple_Bounding_Box_Sequence,self).read_img_into_np_array(img_path,height=self.image_size[0],width=self.image_size[1],tf_transformation=self.data_transformation)
			if img_data is None:
				continue
			if self.preprocess_on_init:
				x = img_data
			self.x.append(x)
			self.y.append(y)
		print('... initialization of data generator finished \t '
				'[generator item number:{} / dropped:{}/{}]'.format(
					len(self.x),
					len(data['train'])-len(self.x),
					len(data['train'])))
		print()

	def __getitem__(self, index):
		'''
		# Returns:
			tuple (X,Y) of data where
				X: [batch_size,img_width,img_height,rgb]
				Y: [batch_size,num_gt_boxes,4]
		'''
		batch_y = self.y[index * self.batch_size:(index+1) * self.batch_size]
		if self.preprocess_on_init:
			batch_x = self.x[index * self.batch_size:(index+1) * self.batch_size]
		else:
			batch_x = []
			for img_path in self.x[index * self.batch_size:(index+1) * self.batch_size]:
				img_data = super(Simple_Bounding_Box_Sequence,self).read_img_into_np_array(img_path,height=self.image_size[0],width=self.image_size[1],tf_transformation=self.data_transformation)
				if img_data is not None:
					batch_x.append(img_data)
		return np.array(batch_x),batch_y

class Detection_Data_Sequence(Keras_Detection_Sequence):
	DEFAULT_SIMILARITY_FUNCTION = 'jaccard_similarity'

	@staticmethod
	def unpack_item(item, label_map):
		data = item['annotation']
		img_path = data['path']
		box_list = []
		if 'object' in data:
			#@TODO implement label map check
			for item_object in data['object']:
				box_list.append([item_object['bndbox']['ycenter'],
					item_object['bndbox']['xcenter'],
					item_object['bndbox']['bheight'],
					item_object['bndbox']['bwidth']])
		return img_path,box_list

	@staticmethod
	def calc_similarity_matrix(box_list_a,box_list_b,similarity_func=DEFAULT_SIMILARITY_FUNCTION,**kwargs):
		try:
			calc_similarity = getattr(similarity,similarity_func)
		except AttributeError:
			calc_similarity = similarity.jaccard_similarity

		similarity_matrix = np.zeros((box_list_a.shape[0],box_list_b.shape[0]),dtype=float)
		for j,ground_truth_box in enumerate(box_list_a):
			for i,anchor_box in enumerate(box_list_b):
				similarity_matrix[j][i] = calc_similarity(ground_truth_box,anchor_box)
		return similarity_matrix

	@staticmethod
	def assign(box_list_a,box_list_b,ignore_boundary_crossings,pos_iou_assign_threshold,neg_iou_assign_threshold,drop_groundtruth_for_iou_lower_than,**kwargs):
		''' Assigns list a of box encodings (e.g.,ground truth)
		to list b of box encodings (e.g., default anchor boxes).
		Only boxes that have at least a defined overlap are assigned.

		Args:
			box_list_a: numpy array of shape [M,4],
				where M is the number of ground truth boxes
			box_list_b: numpy array of shape [N,4],
				where N is the number of anchor boxes
			ignore_boundary_crossings:
			pos_iou_assign_threshold: box a is assigned to box b if similarity
				between both boxes is at least pos_iou_assign_threshold; box b
				gets assigned a positive label in the assignment_matrix
			neg_iou_assign_threshold: if maximum overlap of a box b with any
				box a is less than specified value, then box b gets assigned a
				negative label in the assignment_matrix
		'''
		assert pos_iou_assign_threshold >= neg_iou_assign_threshold, 'positive IoU threshold should be at least as high as negative IoU assignment bound.'
		assignment_matrix = np.zeros((box_list_a.shape[0],box_list_b.shape[0]),dtype=float)
		similarity_matrix = Detection_Data_Sequence.calc_similarity_matrix(box_list_a,box_list_b,**kwargs)

		# ensure anchors which crosses boundaries are ignored during training
		if ignore_boundary_crossings:
			#print('training mode is {}\t IoU for boundary crossing anchors are set to -1'.format(ignore_boundary_crossings))
			for i in range(similarity_matrix.shape[1]):
				if not super(Detection_Data_Sequence,Detection_Data_Sequence).anchor_is_in_img_bounds(*box_list_b[i]):
					similarity_matrix[:,i] = -1.
		assigned = []
		for j in range(similarity_matrix.shape[0]):
			# loop over ground truth
			max_i = np.argmax(similarity_matrix[j,:])
			if similarity_matrix[j,max_i] < drop_groundtruth_for_iou_lower_than:
				print('skip ground truth box beacause max IoU with any anchor is only {}'.format(similarity_matrix[j,max_i]),box_list_a[j])
			else:
				if ignore_boundary_crossings and not super(Detection_Data_Sequence,Detection_Data_Sequence).anchor_is_in_img_bounds(*box_list_b[max_i]):
					print('ERROR 1: {} crosses image bounds'.format(max_i))
				if max_i not in assigned:
					#other_j = np.argmax(similarity_matrix[:,max_i])
					#other_i = np.argmax(np.concatenate((similarity_matrix[j,:max_i],similarity_matrix[j,:max_i+1]),axis=1))
					#print(similarity_matrix[j,other_i],similarity_matrix[other_j,max_i])
					#if similarity_matrix[other_j,max_i] > similarity_matrix[j,max_i]:
					#	similarity_matrix[j,max_i] = -1.
					assignment_matrix[j,max_i] = 1.
					for k in range(similarity_matrix.shape[0]):
						if k != j:
							similarity_matrix[k,max_i] = -1.
					assigned.append(max_i)
					#print('assigned {} to {}'.format(max_i,j))
				else:
					print('ERROR: {} is already assigned'.format(max_i))
		for i in range(similarity_matrix.shape[1]):
			assignment_test = np.max(assignment_matrix[:,i])
			if i not in assigned and (not ignore_boundary_crossings or super(Detection_Data_Sequence,Detection_Data_Sequence).anchor_is_in_img_bounds(*box_list_b[i])):
				assert assignment_test <= 0.
				max_j = np.argmax(similarity_matrix[:,i])
				if similarity_matrix[max_j,i] >= pos_iou_assign_threshold:
					if not super(Detection_Data_Sequence,Detection_Data_Sequence).anchor_is_in_img_bounds(*box_list_b[i]) and ignore_boundary_crossings:
						print('ERROR 2: {} crosses image bounds'.format(i))
					assignment_matrix[max_j,i] = 1.
					#print('assigned {} to {}'.format(i,max_j))
				elif similarity_matrix[max_j,i] < neg_iou_assign_threshold:
					if not super(Detection_Data_Sequence,Detection_Data_Sequence).anchor_is_in_img_bounds(*box_list_b[i]) and ignore_boundary_crossings:
						print('ERROR 3: {} crosses image bounds'.format(i))
					assignment_matrix[max_j,i] = -1.
					pass
			else:
				# case i should be ignored due to image boundary violations or i is assigned
				try:
					assert assignment_test == 1.
				except:
					try:
						assert not super(Detection_Data_Sequence,Detection_Data_Sequence).anchor_is_in_img_bounds(*box_list_b[i])
					except:
						print(i)
						print(ignore_boundary_crossings)
						print(super(Detection_Data_Sequence,Detection_Data_Sequence).anchor_is_in_img_bounds(*box_list_b[i]))
						print(assigned)
						print(assignment_matrix[:,i])
						print(similarity_matrix[:,i])
						print(similarity_matrix.shape)
						print(assignment_matrix.shape)
						raise Exception()

		try:
			assert np.all(np.amax(assignment_matrix,axis=0)*np.amin(assignment_matrix,axis=0)==0.), 'anchor box assignment should not be ambigious for truth box: {}'.format(box_list_a)
		except AssertionError:
			assert assignment_matrix.shape[0] == 1, 'anchor box assignment should not be ambigious'
		try:
			assert np.sum(np.amax(assignment_matrix,axis=1)) == box_list_a.shape[0], 'not all ground truth boxes are assigned to an anchor'
		except:
			assert drop_groundtruth_for_iou_lower_than > 0.
		assert np.unique(assignment_matrix).shape[0] <= 3, 'more labels than [-1,0,1] assigned'
		assert len(assignment_matrix.shape)==2, 'assignment matrix should be 2 dimensional'
		return assignment_matrix

	@staticmethod
	def generate_anchor_targets(ground_truth_box_list,anchor_box_list,ignore_boundary_crossings,**kwargs):
		assignment_matrix = Detection_Data_Sequence.assign(ground_truth_box_list,anchor_box_list,ignore_boundary_crossings,**kwargs)
		labels = assignment_matrix[0]
		if assignment_matrix.shape[0] > 1:
			labels = np.amax(assignment_matrix,axis=0)+np.amin(assignment_matrix,axis=0)

		targets = np.concatenate((anchor_box_list,np.expand_dims(labels,axis=1)),axis=1)
		for j in range(assignment_matrix.shape[1]):
			max_i = np.argmax(assignment_matrix[:,j])
			if assignment_matrix[max_i,j] == 1.:
				#targets[j,:4] = ground_truth_box_list[max_i]
				targets[j,:4] = super(Detection_Data_Sequence,Detection_Data_Sequence).generate_regression_targets(targets[j,:4],ground_truth_box_list[max_i])
			else:
				# anchor is either unassigned or negative
				targets[j,:4] = np.zeros((4,),dtype=float)
		return targets

	'''
	@staticmethod
	def y_true_func(ground_truth_boxes,linear_anchor_array,grid_sizes_list,num_anchors_per_location,ignore_boundary_crossings,**kwargs):
		linear_target_array = Detection_Data_Sequence.generate_anchor_targets(
			ground_truth_boxes,
			linear_anchor_array,
			ignore_boundary_crossings,
			**kwargs)
		return super(Detection_Data_Sequence,Detection_Data_Sequence)._fit_to_model_outputs(linear_target_array)
	'''

	@classmethod
	def test(cls):
		print(cls)
		print(DEFAULT_SIMILARITY_FUNCTION)
		print(cls.DEFAULT_SIMILARITY_FUNCTION)

	def __init__(self,data,ignore_boundary_crossings=True,label_map={},pos_iou_assign_threshold=0.5,neg_iou_assign_threshold=0.4,drop_groundtruth_for_iou_lower_than=GT_IOU_DROP_UNDER,similarity_func=DEFAULT_SIMILARITY_FUNCTION,**kwargs):
		'''
		Args:
			pos_iou_assign_threshold
		'''
		super(Detection_Data_Sequence,self).__init__(**kwargs)
		self.x = []
		self.y = [] # list of M lists where M is number of feature layers
		self.anchor_kwargs = {
			'ignore_boundary_crossings':ignore_boundary_crossings,
			'label_map':label_map,
			'pos_iou_assign_threshold':pos_iou_assign_threshold,
			'neg_iou_assign_threshold':neg_iou_assign_threshold,
			'drop_groundtruth_for_iou_lower_than':drop_groundtruth_for_iou_lower_than,
			'similarity_func':similarity_func,
		}
		self.ignore_boundary_crossings=ignore_boundary_crossings
		self.label_map=label_map
		self.pos_iou_assign_threshold=pos_iou_assign_threshold
		self.neg_iou_assign_threshold=neg_iou_assign_threshold
		self.drop_groundtruth_for_iou_lower_than=drop_groundtruth_for_iou_lower_than
		self.similarity_func=similarity_func

		'''
			self.batch_size = batch_size
			self.image_size = image_size
			self.preprocess_on_init = preprocess_on_init
			if preprocess_on_init:
				print('read image at init time')
			else:
				print('read image at batch load time')
			self.grid_sizes = grid_sizes
			if aspect_ratios:
				print('generate anchor encodings for provided aspect_ratios:\t',aspect_ratios)
				self.aspect_ratios = aspect_ratios
			else:
				print('no aspect ratios provided, generate default encodings:\t',self.ASPECT_RATIOS)
				self.aspect_ratios = self.ASPECT_RATIOS

			self.min_scale = min_scale
			self.max_scale = max_scale

			self.linear_anchor_array = super(Detection_Data_Sequence,self).generate_anchor_list(
				grid_sizes,
				aspect_ratios,
				min_scale,
				max_scale)

			self.num_anchors_per_location = super(Detection_Data_Sequence,self)._anchors_per_grid_cell(aspect_ratios)
		'''

		if data is not None:
			print('processing data ...')
			for item_num,item in enumerate(data):
				if (item_num+1) % 500 == 0:
					print('processing item {}/{} ...'.format(item_num+1,len(data)))
				img_path,ground_truth_boxes = self.unpack_item(item, label_map)
				if not ground_truth_boxes:
					continue

				filename = item['annotation']['filename']
				download_folder = kwargs.get('img_download_dir',os.path.join('/','tmp'))
				tmp_img_path = os.path.join(download_folder,filename)
				if os.path.exists(tmp_img_path) and os.path.isfile(tmp_img_path):
					img_path = tmp_img_path

				x = img_path
				y = np.array(ground_truth_boxes)
				if urllib.parse.urlparse(img_path).scheme != "":
					#if not os.path.exists(tmp_filepath) or not os.path.isfile(tmp_filepath):
					try:
						response = requests.get(img_path)
						if response.status_code == 200:
							img = Image.open(BytesIO(response.content))
							img = img.resize((self.image_size[1], self.image_size[0]))
							img.save(tmp_filepath)
							x = tmp_img_path
							#print('downloaded image from {} and stored at {}'.format(img_path,tmp_filepath))
						else:
							raise Exception('could not open url({}): {}'.format(img_path,response.status_code))
					except Exception as e:
						print(e)
						continue
					item['annotation']['path'] = x

				img_data = super(Detection_Data_Sequence,self).read_img_into_np_array(img_path,height=self.image_size[0],width=self.image_size[1],tf_transformation=self.data_transformation)
				if img_data is None:
					continue
				print('loaded item into memory:\t{}'.format(img_path))

				# transform ground truth to y
				linear_target_array = self.generate_anchor_targets(
					y,
					self.linear_anchor_array,
					**self.anchor_kwargs)
				y_true_list = self.fit_to_model_outputs(linear_target_array)

				pos = [0 for i in range(len(y_true_list))]
				ngs = [0 for i in range(len(y_true_list))]
				nas = [0 for i in range(len(y_true_list))]
				for feat_map_index,y_true_numpy_map in enumerate(y_true_list):
					for row in y_true_numpy_map:
						for cell in row:
							for anchor in cell:
								if anchor[-1] == 1.:
									pos[feat_map_index]+=1
								elif anchor[-1] == -1.:
									ngs[feat_map_index]+=1
								else:
									nas[feat_map_index]+=1 # ignored by boundary violatin or threshold inbetweenness
				if np.sum(pos) == 0:
					print('drop item due to 0 ground truth assignments')
					print(np.sum(pos))
					print(pos,ngs,nas,img_path)
					print()
					continue

				if self.preprocess_on_init:
					x = img_data
					'''
					for feat_map_index,y_true_numpy_map in enumerate(y_true_list):
						if len(self.y) <= feat_map_index:
							# append list for feature map/output to y
							self.y.append(list())
						self.y[feat_map_index].append(y_true_numpy_map)
					'''
					y = y_true_list
					self.y.append(y)
				else:
					#derive file path from x
					prefix,_ = os.path.splitext(x)
					x = '{}.npy'.format(prefix)
					print('save data to {}'.format(x))
					np.save(x,np.array([img_data,y_true_list]))
				self.x.append(x)
		else:
			print('no init data provided')
		print('... initialization of data generator finished \t [generator item number:{} / dropped:{}/{}]'.format(len(self.x),len(data)-len(self.x),len(data)))
		print()

	def __getitem__(self, index):
		if self.preprocess_on_init:
			batch_x = self.x[index * self.batch_size:(index+1) * self.batch_size]
			y_slice = self.y[index * self.batch_size:(index+1) * self.batch_size]
		else:
			numpy_files_x = self.x[index * self.batch_size:(index+1) * self.batch_size]
			batch_x = []
			y_slice = []
			for file in numpy_files_x:
				data = np.load(file)
				batch_x.append(data[0])
				y_slice.append(data[1])
			'''
			for img_path in self.x[index * self.batch_size:(index+1) * self.batch_size]:
				img_data = super(Detection_Data_Sequence,self).read_img_into_np_array(img_path,height=self.image_size[0],width=self.image_size[1])
				batch_x.append(img_data)
			'''

			'''
			i = 0
			start_index = index * self.batch_size
			while i < self.batch_size:
				try:
					img_path_i = self.x[start_index+i]
					y_i = self.y[start_index+i]
				except:
					break
				img_data_i = super(Detection_Data_Sequence,self).read_img_into_np_array(img_path_i,height=self.image_size[0],width=self.image_size[1])
				linear_target_array = self.generate_anchor_targets(
					y_i,
					self.linear_anchor_array,
					**self.anchor_kwargs)
				y_true_list = self.fit_to_model_outputs(linear_target_array)
				if img_data_i is not None:
					batch_x.append(img_data_i)
					y_slice.append(y_true_list)
					i += 1
				else:
					before_length = len(self.x)
					if start_index+i+1 < len(self.x):
						self.x = self.x[:start_index+i]+self.x[start_index+i+1:]
						self.y = self.y[:start_index+i]+self.y[start_index+i+1:]
					print('item {} not valid, delete it from list ({}>{})'.format(img_path_i,before_length,len(self.x)))
			for y in self.y[index * self.batch_size:(index+1) * self.batch_size]:
				linear_target_array = self.generate_anchor_targets(
					y,
					self.linear_anchor_array,
					**self.anchor_kwargs)
				y_true_list = self.fit_to_model_outputs(linear_target_array)
				y_slice.append(y_true_list)
			'''

		batch_y = []
		for item_index,y_true_map_list in enumerate(y_slice):
			for feat_map_index,y_true_numpy_map in enumerate(y_true_map_list):
				if len(batch_y) <= feat_map_index:
					batch_y.append(np.empty((0,*y_true_numpy_map.shape)))
				exp_y_true_numpy_map = np.expand_dims(y_true_numpy_map,axis=0)
				batch_y[feat_map_index] = np.append(batch_y[feat_map_index],exp_y_true_numpy_map,axis=0)
		'''
		for dim_items in self.y:
			batch_y_i = dim_items[index * self.batch_size:(index+1) * self.batch_size]
			batch_y.append(np.array(batch_y_i))
		'''
		return np.array(batch_x),batch_y
