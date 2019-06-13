'''
Script to evaluate predictions with mAP and custom metrics.
The script can be executed in detection_only mode which means that no
annotations are required and only images can be passed to the image.
In detection_only mode no metric is calculated but detections are plotted
and stored in the directory ./data/detection_output/

# Execution emaple()
python eval_model.py \
	--model_dir=models/tensorflow_models/ssd_inception_v2_training_2/ \
	--input_data_dir=data/images/detection_test/ \
	--iou_boundary=0.4 \
	--confidence_threshold=0.3 \
	--detection_only \
	--tensorflow_model

python eval_model.py \
	--model_dir=models/keras_models/Lfm_SeparatedParams_drop_0.5/ \
	--input_data_dir=data/images/detection_test/ \
	--iou_boundary=0.4 \
	--confidence_threshold=0.3 \
	--detection_only

python eval_model.py \
	--model_dir=models/tensorflow_models/ssd_inception_v2_training_2/ \
	--input_data_dir=data/trainings/graffiti_ba_training/annotations/test \
	--iou_boundary=0.5 \
	--confidence_threshold=0.3 \
	--tensorflow_model

python eval_model.py \
	--model_dir=models/keras_models/Lfm_SeparatedParams_drop_0.5/ \
	--input_data_dir=data/trainings/graffiti_ba_training/annotations/test \
	--iou_boundary=0.5 \
	--confidence_threshold=0.3 \
	--preprocess_on_init \
	--output_flag


# Args:
	-m --model_dir path to model directory
	-w --weights_file_path (optional) path to exported checkpoint file
	-d --input_data_dir path to directory containing images or xml markup files
	-p --pipeline_path (optional/only for keras models) path to model pipeline config file
	-n --max_number_of_testfiles (optional/unlimited if not provided) process only limited amount of test items
	-u --iou_boundary (optional/default:0.4) IoU value at which a prediction should be treated as true positive
	-c --confidence_threshold (optional/default:0.6) only consider predictions with at least certain confidence scores
	-q --max_detections_per_image (optional/default:10) set a limit for the max number of detections per image
	
	-i --preprocess_on_init (optional/default:false) flag only parameter indicates whether data should be preprocessed at initialization time
	-b --downrate_off_bounds (optional/default:false) flag onlu parameter if set, predictions that cross image bounds are ignored
	-t --tensorflow_model flag only parameter indicates that tensorflow object detection api should be used to generate predictions
	-a --detection_only only generates detections, this mode does not require any annotated data, image directory as --input_data_dir is enough
	-o --output_flag (optional/default:false) flag only parameter indicates whether detections should be plotted to images and saved to disk in evaluation mode 
'''
import os
import sys
import re
import getopt
import numpy as np
from utils import console_utils, data_sequence, losses, extractAnnotations, similarity, classification_auxiliary
import train_inceptionV3_base, generateAnnotationFromModelsPredictions
from object_detection.utils import visualization_utils
from keras.preprocessing.image import save_img, img_to_array, load_img

PARAMS = [
	[
		("m","model_dir",),
		("w","weights_file_path",),
		("d","input_data_dir",),
		("p","pipeline_path",),
		("n","max_number_of_testfiles",),
		("u","iou_boundary",),
		("c","confidence_threshold",),
		("q","max_detections_per_image",),
		],
	[
		('i','preprocess_on_init',),
		('b','downrate_off_bounds',),
		('t','tensorflow_model',),
		('a','detection_only',),
		('o','output_flag',),
		],
]

IOU_BOUNDARY = 0.5
CONFIDENCE_THRESHOLD = 0.6
MAX_DETECTIONS = 10
DEFAULT_IMG_SIZE = 460
DEFAULT_OUTPUT_SUBDIR = 'data'
EVAL_DATA_SUBDIR_NAME = 'evaluation_output'
DETECT_DATA_SUBDIR_NAME = 'detection_output'
	
def generate_kwargs_dict(opts,args):
	'''
	Returns:
		kwargs: dictionary containing the following fields
			model_dir: path to the model directory
			markup_path: directory holding the annotation files
			keras_preprocessing_mode: boolean indicating if model_dir 
				contains keras model (default:true)
			weights_ckpt_file: list [selected] with paths to model checkpoints
			preprocess_on_init: boolean indicates if data sequence should be
				processed on init (only relevant for keras models)
			max_data_size: number of maximum data items to process
			prediction_mode: boolean indicating whether the method should
				output predictions only (true) or evaluate predictions (false)
			output_flag: boolean indicating whether the predictions should
				be plotted to images in evaluation mode
			iou_overlap_threshold: float between 0 and 1 indicating the minimum
				overlap	between prediction and ground truth at which the 
				prediction should be labeled as true positive
			max_number_of_detections:
			downrate_out_off_bounds: bool if true the confidence of all
				prediction that crosses an image boundary is set to 0 which
				means that these predictions are ignored for eval/detection
			similarity_func: function that should be used to calculate IoU 
				between bounding boxes (default:jaccard)
			pos_iou_assign_threshold: threshold for labeling a prediction as TP
			iou_overlap_threshold: threshold at which corresponding detections 
				are considered to detect the same ground truth object
	'''
	kwargs = {}

	model_dir = console_utils.extract_option(PARAMS[0][0],opts)
	if model_dir is None:
		raise EnvironmentError('path to model dir must be provided')
	else:
		kwargs['model_dir'] = model_dir

	test_data_root = console_utils.extract_option(PARAMS[0][2],opts)
	if test_data_root is None:
		raise EnvironmentError('path to test data must be provided')
	else:
		kwargs['markup_path'] = test_data_root

	kwargs['keras_preprocessing_mode'] = not console_utils.check_flag_presence(PARAMS[1][2],opts)
	
	weights_ckpt = console_utils.extract_option(PARAMS[0][1],opts)
	if weights_ckpt is None:
		if kwargs['keras_preprocessing_mode']:
			weights = train_inceptionV3_base.list_model_weights(model_dir)
		else:
			weights = train_inceptionV3_base.list_tf_checkpoints(model_dir)
		
		selected = [] # hold indices into weights list
		if len(weights)>1:
			while True:
				print('Please select checkpoint for evaluation:')
				for i,weights_ckpt_cand in enumerate(weights):
					if i in selected:
						print('{:d} [*]\t{}'.format(i+1,weights_ckpt_cand))
					else:
						print('{:d}\t{}'.format(i+1,weights_ckpt_cand))
				print('Input selected number [{}:{}]'.format(1,len(weights)))
				x = input()
				try:
					select = int(x)
					if 0 < select <= len(weights):
						item = select-1
						if item in selected:
							selected = selected[:selected.index(item)]+selected[selected.index(item)+1:]
						else:
							selected.append(item)
					else:
						break
				except:
					break
		else:
			ckpt_file = os.path.join(model_dir,weights[0])
			selected.append(0)
			print('Only one checkpoint found, use {}'.format(ckpt_file))
		if not selected:
			raise Exception('no weight file selected')
		weights_ckpt = []
		print('selected checkpoint(s):')
		for i,select in enumerate(selected):
			print('-\t{}'.format(weights[select]))
			weights_ckpt.append(os.path.join(model_dir,weights[select]))
		print()
	kwargs['weights_ckpt_file'] = weights_ckpt
	if not isinstance(kwargs['weights_ckpt_file'],list):
		kwargs['weights_ckpt_file'] = [kwargs['weights_ckpt_file']]

	if kwargs['keras_preprocessing_mode']:
		pipeline_path = console_utils.extract_option(PARAMS[0][3],opts)
		if pipeline_path is None:
			pipeline_path = os.path.join(kwargs['model_dir'],'pipeline.yaml')
			if not os.path.isfile(pipeline_path):
				raise Exception('path to pipeline file must be provided')
			print('use default pipeline.yaml from model directory')


		config = train_inceptionV3_base.load_config(pipeline_path)	
		kwargs['aspect_ratios'] = config['detection_config']['aspect_ratios']
		kwargs['min_scale'] = config['detection_config']['min_scale']
		kwargs['max_scale'] = config['detection_config']['max_scale']
		kwargs['max_anchor_limit'] = config['detection_config']['max_anchor_limit']
		kwargs['batch_size'] = config['detection_config']['batch_size']
		#kwargs['pos_iou_assign_threshold'] = config['detection_config']['pos_iou_assign_threshold']
	
	kwargs['similarity_func'] = 'jaccard_similarity'
	
	kwargs['preprocess_on_init'] = False
	if kwargs['keras_preprocessing_mode']:
		kwargs['preprocess_on_init'] = console_utils.check_flag_presence(PARAMS[1][0],opts)

	kwargs['max_data_size'] = -1
	max_data_size = console_utils.extract_option(PARAMS[0][4],opts)
	if max_data_size is not None:
		kwargs['max_data_size'] = int(max_data_size)
	
	kwargs['prediction_mode'] = console_utils.check_flag_presence(PARAMS[1][3],opts)
	kwargs['output_flag'] = console_utils.check_flag_presence(PARAMS[1][4],opts)

	kwargs['iou_overlap_threshold'] = IOU_BOUNDARY
	positive_IoU_boundary = console_utils.extract_option(PARAMS[0][5],opts)
	if positive_IoU_boundary is not None:
		kwargs['iou_overlap_threshold'] = float(positive_IoU_boundary)
	kwargs['pos_iou_assign_threshold'] = kwargs['iou_overlap_threshold']
		

	kwargs['confidence_threshold'] = CONFIDENCE_THRESHOLD
	confidence_threshold = console_utils.extract_option(PARAMS[0][6],opts)
	if confidence_threshold is not None:
		kwargs['confidence_threshold'] = float(confidence_threshold)
	
	kwargs['max_number_of_detections'] = MAX_DETECTIONS
	max_number_of_detections = console_utils.extract_option(PARAMS[0][7],opts)
	if max_number_of_detections is not None:
		kwargs['max_number_of_detections'] = int(max_number_of_detections)

	kwargs['downrate_out_off_bounds'] = console_utils.check_flag_presence(PARAMS[1][1],opts)
	return kwargs

def generate_model(model_dir,weights_ckpt_file=None,**kwargs):
	''' Load a keras model from a config file and init
	weights from saved checkpoint.

	Args:
		model_dir: directory that includes the model's `model_config.yaml`
		weights_ckpt_file: either list or string of paths to checkpoint files
			only first checkpoint from a list gets loaded

	Returns:
		model: instance of a keras model
	'''
	print('generating model:')
	model = train_inceptionV3_base.model_from_yaml(model_dir)
	
	if weights_ckpt_file is None:
		print('-\tno checkpoint loaded')
	else:
		if isinstance(weights_ckpt_file,list):
			weights_ckpt = weights_ckpt_file[0]
		else:
			weights_ckpt = weights_ckpt_file
		
		print('-\tloading checkpoint: {}'.format(weights_ckpt_file.split('/')[-1]))
		model.load_weights(weights_ckpt)
	print('[OK]\tmodel is ready\n')
	return model

def detect(predictions,iou_overlap_threshold,confidence_threshold,max_number_of_detections,max_number_of_considered_predictions=200,**kwargs):
	''' Calculates the final detections from a list of predictions according
	to some constraints.

	Args:
		predictions: [num_predictions,5] linear numpy array 
			of predictions sorted by confidence score in 
			decreasing order (i.e., the most confident 
			predictions first)
		iou_overlap_threshold: ignore predictions that have an
			overlap of at least this value with any other prediction
			that has already been considered
		confidence_threshold: only predictions with confidence
			score of at least the threshold value are considered
			as possible positive detections
		max_number_of_detections: upper boundary for number of
			detections
		max_number_of_considered_predictions: upper boundary for
			number of predicitons that should be considered as 
			possible detections

	Returns:
		positive_detection_indices: [num_detections] indices into predictions
		positive_detections: [num_detections,5] prediction encodings
	'''
	#if isinstance(max_number_of_considered_predictions,int):
	#	max_number_of_considered_predictions = np.array([max_number_of_considered_predictions]*predictions.shape[0])
	for i,prediction in enumerate(predictions):
		#if prediction[-1] < confidence_threshold or i >= max_number_of_considered_predictions[i]:
		if prediction[-1] < confidence_threshold or i >= max_number_of_considered_predictions:
			break

	predictions_overlap_matrix = data_sequence.Detection_Data_Sequence.calc_similarity_matrix(predictions[:i],predictions[:i],**kwargs)
	positive_detection_indices = np.empty((0,),dtype=int)
	positive_detections = np.empty((0,*predictions.shape[1:]))
	for k in range(i):
		if positive_detection_indices.shape[0] == 0:
			positive_detection_indices = np.append(positive_detection_indices,k)
			positive_detections = np.concatenate((positive_detections,np.expand_dims(predictions[k],axis=0)),axis=0)
		else:
			overlap = False
			for j in positive_detection_indices:
				if predictions_overlap_matrix[k][j] >= iou_overlap_threshold:
					overlap = True
					break
			if not overlap:
				positive_detection_indices = np.append(positive_detection_indices,k)
				positive_detections = np.concatenate((positive_detections,np.expand_dims(predictions[k],axis=0)),axis=0)
		if positive_detection_indices.shape[0] >= max_number_of_detections > 0:
			print('\tmax number of detections per image ({}) reached, stop generating detections...'.format(max_number_of_detections))
			break
	return positive_detection_indices,positive_detections

def label_predictions(predictions,ground_truth,iou):
	''' Assigns a label, either true positive (>=0) or false positive (<0)
	to each prediction according to the IoU with ground truth labels.

	Args:
		predictions: [num_predictions,5] linear numpy array 
			of predictions sorted by confidence score in 
			decreasing order (i.e., the most confident 
			predictions first)
		ground_truth: [num_ground_truth_objects,4] numpy array
			of ground truth boxes
		iou: minimume overlap at which a prediction is 
			considered true positive

	Returns:
		predictions: [num_predictions] labels
		last_positive_index: index into predictions where recall is 1
			(i.e., all ground truth objects have been assigned to a
			prediction)
	'''
	similarity_matrix = data_sequence.Detection_Data_Sequence.calc_similarity_matrix(ground_truth,predictions,**kwargs)
	prediction_labels = np.array([-1.]*predictions.shape[0])
	last_positive_index = -1
	gt_duplicates = []
	for i in range(similarity_matrix.shape[1]):
		max_j = np.argmax(similarity_matrix[:,i])
		if max_j not in gt_duplicates and similarity_matrix[max_j,i] >= iou:
			# assign 1 (the most confident) prediction to each ground truth object
			gt_duplicates.append(max_j)
			prediction_labels[i] = max_j
		if len(gt_duplicates) == similarity_matrix.shape[0]:
			break
	last_positive_index = i
	#assert len(gt_duplicates) == similarity_matrix.shape[0], 'no overlap for {} ground truth objects found with {} predictions'.format(similarity_matrix.shape[0]-len(gt_duplicates),similarity_matrix.shape[1])
	return prediction_labels,last_positive_index

def cals_precision_recall_values(prediction_labels,num_ground_truth_items):
	'''Calculate precision and recall values for a series of predicitons.
	
	Args:
		prediciton_labels: [num_predictions] labels of the predicitons
		num_ground_truth_items: number of total ground truth objects

	Returns:
		numpy array [num_prediction,2] with precision and recall value
			for each prediction 
	'''
	precision = np.zeros(prediction_labels.shape)
	recall = np.zeros(prediction_labels.shape)
	tp = 0
	fp = 0
	for i in range(prediction_labels.shape[0]):
		if prediction_labels[i] >= 0:
			tp += 1
		else:
			fp += 1
		precision[i] = tp / (tp+fp)
		recall[i] = tp / num_ground_truth_items
	return np.concatenate((np.expand_dims(precision,axis=1),np.expand_dims(recall,axis=1)),axis=1)

def average_precision(predictions,ground_truth,pos_iou_assign_threshold,**kwargs):
	''' Calcualtes three average precision values and the fraction of
	found ground truth objects.

	Args:
		predictions: [num_predictions,5] linear numpy array 
			of predictions sorted by confidence score in 
			decreasing order (i.e., the most confident 
			predictions first)
		ground_truth: [num_ground_truth_objects,4] numpy array
			of ground truth boxes
		pos_iou_assign_threshold

	Returns:
		ap: average precision over all true positive points
		distinct_ap: average precision over 11 distinct recall points (mAP version)
		steady_ap: average precision over all predictions until recall = 1
		coverage: ratio of true positives to ground truth objects
	'''
	prediction_labels,last_positive_index = label_predictions(predictions,ground_truth,pos_iou_assign_threshold)

	metric_values = cals_precision_recall_values(prediction_labels,ground_truth.shape[0])
	
	detected_items = np.count_nonzero(prediction_labels >= 0)
	coverage = np.divide(detected_items,ground_truth.shape[0])

	recalls = [0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.]
	precision_accum = 0.
	for recall in recalls:
		max_ap_i = 0.
		indices = np.where(metric_values[:,1]>=recall)[0]
		if indices.shape[0] > 0:
			recall_i = indices[0]
			max_ap_i = np.amax(metric_values[recall_i:,0])
		precision_accum += max_ap_i
	distinct_ap = np.divide(precision_accum,len(recalls))

	precision_accum = 0.
	for i in range(metric_values.shape[0]):
		precision_accum += metric_values[i][0]
		if metric_values[i][1] >= 1.:
			# recall = 1, i.e. all true positives found
			break
	steady_ap = np.divide(precision_accum,i+1) 

	precision_accum = 0.0
	ranked_items = 0
	for i in range(prediction_labels.shape[0]):
		if prediction_labels[i] >= 0:
			precision_accum += metric_values[i,0]
			ranked_items += 1
		if i == last_positive_index:
			break
	ap = 0.
	if ranked_items > 0:
		ap = np.divide(precision_accum,ranked_items)
	return ap,distinct_ap,steady_ap,coverage

def score_until_negative_hit(per_prediction_items,number_of_different_gt_objects,pos_iou_assign_threshold,**kwargs):
	''' Implementation of the custom metrics.
	Counts the number of true positives before the 
	first false positive prediction occurs in a list 
	of sorted predictions.

	Args:
		per_prediction_items: [num_predictions,3] where values are
			- 0: index of ground truth object with max overlap
			- 1: IoU with that ground truth object 
			- 2: confidence score
		number_of_different_gt_objects: num_ground_thruth_obj
		pos_iou_assign_threshold: threshold for TP labelling

	Returns:
		objects_found: [found] indices of found ground truth objetes
		summed_confidence: accumulated confidence of TP predictions
			that are reflected by metric
		summed_iou: accumulated iou of TP predictions
			that are reflected by metric
		neg_conf: accumulated confidence of all FP predictions
		negs: number of FP predictions
		neg_conf_inv: accumulated confidence of all TP predictions
		negs_inv: number of TP predictions
	'''
	objects_found = []
	summed_confidence = 0.0
	summed_iou = 0.0
	neg_conf = 0.0
	negs = 0
	neg_conf_inv = 0.0
	negs_inv = 0
	for i in range(per_prediction_items.shape[0]):
		prediction = per_prediction_items[i]
		if len(objects_found) >= number_of_different_gt_objects:
			break
		if prediction[1] <= pos_iou_assign_threshold:
			break
		if prediction[0] not in objects_found:
			objects_found.append(prediction[0])
			summed_confidence += prediction[2]
			summed_iou += prediction[1]
	for prediction in per_prediction_items:
		if prediction[1] <= pos_iou_assign_threshold:
			neg_conf += prediction[2]
			negs += 1
		else:
			neg_conf_inv += prediction[2]
			negs_inv += 1
	return np.array(objects_found),summed_confidence,summed_iou,neg_conf,negs,neg_conf_inv,negs_inv

def evaluate(predictions,ground_truth,**kwargs):
	''' Calculates IoU overlap between predictions and 
	ground truth and assigns ground truth items to predictions
	according to maximum IoU overlap. Handles assignment to custom
	metric method.
	
	Args:
		predictions: [num_predictions,5] linear list of 
			predictions, sorted in decreasing order along
			confidence score
		ground_truth: [num_ground_thruth_obj,4] list of
			ground truth data

	Returns: @see score_until_negative_hit
	'''
	similarity_matrix = data_sequence.Detection_Data_Sequence.calc_similarity_matrix(ground_truth,predictions,**kwargs)

	# save ground truth index,overlap,confidence for ground truth boxes j which has highest overlap with prediction item i
	per_prediction_items = np.empty((similarity_matrix.shape[1],3))
	for k in range(similarity_matrix.shape[1]):
		max_j = np.argmax(similarity_matrix[:,k])
		iou_j = similarity_matrix[max_j,k]
		ci = predictions[k,-1]
		per_prediction_items[k,0] = max_j
		per_prediction_items[k,1] = iou_j
		per_prediction_items[k,2] = ci

	return score_until_negative_hit(
			per_prediction_items,
			ground_truth.shape[0],
			**kwargs)
				
def plot_box_to_img(img_np,min_max_box,color='red',thickness=4,text=''):
	''' Plots an anchor box to image.

	Args:
		img_np: [height,width,rgb] image data
		min_max_box: [ymin,ymax,xmin,xmax] bounding box encoding
		color: of the box borders
		thickness: of the box borders
		text: label of the box
	'''
	ymin = min_max_box[0]
	ymax = min_max_box[2]
	xmin = min_max_box[1]
	xmax = min_max_box[3]
	if ymin < 0. and ymax > 1. and xmin < 0. and xmax > 1.:
		print('box completly out of img')
	visualization_utils.draw_bounding_box_on_image_array(
		img_np,
		ymin,
		xmin,
		ymax,
		xmax,
		color=color,
		thickness=thickness,
		display_str_list=[text],
		#display_str_list=('{}%'.format(int(preds[k][4]*100))),
		use_normalized_coordinates=True)

def plot_preds_to_img(img_np,predictions,max_number_of_detections,color='red',thickness=4,plot_conf_label=False,**kwargs):
	''' Plots a set of predictions to image.

	Args:
		img_np: [height,width,rgb] image data
		predictions: [num_predictions,5] list of bounding box encodings
		color: of the box borders
		thickness: of the box borders
		plot_conf_label: boolean if true, confidence score is 
			plotted as box label, otherwise label is blank (default:False)
	'''
	for i,prediction in enumerate(predictions):
		if i >= max_number_of_detections:
			break
		min_max_preds = data_sequence.Keras_Detection_Sequence.convert_norm_to_min_max(prediction[:4])
		conf_value = ''
		if plot_conf_label:
			conf_value = '{:.2f}'.format(prediction[4])
		plot_box_to_img(img_np,min_max_preds,color,thickness,conf_value)

def plot_to_img(img_np,box_indices,box_list,color='red',thickness=4,plot_conf_label=False):
	''' Plots a set of bounding boxes, indicated by index.

	Args:
		img_np: [height,width,rgb] image data
		box_indices: [to_be_printed] indices into box_list
		box_list: [num_predictions]
		color: of the box borders
		thickness: of the box borders
		plot_conf_label: boolean if true, confidence score is 
			plotted as box label, otherwise label is blank (default:False)
	'''
	for box_i in box_indices:
		if box_i < 0:
			continue
		box_i = int(box_i)
		min_max_preds = data_sequence.Keras_Detection_Sequence.convert_norm_to_min_max(box_list[box_i][:4])
		conf_value = ''
		if plot_conf_label:
			conf_value = '{:.2f}'.format(box_list[box_i,4])
			#conf_value = conf_value.encode('utf-8')
		plot_box_to_img(img_np,min_max_preds,color,thickness,conf_value)

def output_eval_result(ckpt,meanAP,dAP,sAP,items,found,total,conf,iou,negs,neg_conf_sum,neg_conf_inv,negs_inv,downrate_out_off_bounds,pos_iou_assign_threshold,model_dir='',coverage=0.,**kwargs):
	''' Outputs final evaluation results and writes final results
	to eval.txt file in model directory.
	'''
	out = (
		'{}\n'
		'downrate out off bounds predictions\t[{}]\n'
		'TP label @IoU\t[{}]\n'
		'---------------------------------\n'
		'[{}%]\tmAP (only at recall change points)\n'
		'[{}%]\tmAP (PVOC/COCO metrics)\n'
		'[{}%]\tmAP (over all predictions until recall=1)\n'
		'[{}%]\tmAP TP coverage of GT\n'
		'---------------------------------\n'
		'[{}]\tcustom #TP before 1st FP\n'
		'[{}]\tcustom #GT\n'
		'[{}%]\tcustom TP coverage of GT\n'
		'[{}%]\tcustom avg. TP confidence\n'
		'[{}%]\tcustom avg. TP/GT IoU\n'
		'=================================\n\n\n'
		.format(
			ckpt,
			downrate_out_off_bounds,
			pos_iou_assign_threshold,
			int(np.round(np.divide(meanAP,items)*100,decimals=0)),
			int(np.round(np.divide(dAP,items)*100,decimals=0)),
			int(np.round(np.divide(sAP,items)*100,decimals=0)),
			int(np.round(np.divide(coverage,items)*100,decimals=0)),
			found,
			total,
			int(np.round(np.divide(found,total)*100,decimals=0)),
			int(np.round(np.divide(conf,found)*100,decimals=0)),
			int(np.round(np.divide(iou,found)*100,decimals=0)),
			)
		)
	print(out)
	with open(os.path.join(model_dir,'eval.txt'),'a') as f:
		f.write(out)

def print_result_state(found,total,meanAP,dAP,sAP,items,coverage,iou):
	''' Outputs current evaluation results.
	'''
	print('\n- currently {} ground truth objects evaluated'.format(
		total,)
	)
	
	print('- custom metric coverage (#TP): {}% ({})'.format(
		int(np.round(found/total*100,decimals=0)),
		found,
		),
		end='',
	)

	print('\tmAP metric coverage (#TP) {}% ({})'.format(
		int(np.round(coverage/items*100,decimals=0)),
		int(np.round(coverage/items*total,decimals=0)),
		),
	)
	
	print('- mAP(@{}):\n'
		'-\t{}% (only at recall change points)\n'
		'-\t{}% (PVOC/COCO metrics)\n'
		'-\t{}% (over all predictions until recall=1)'.format(
			iou,
			int(np.round((meanAP/items*100),decimals=0)),
			int(np.round((dAP/items*100),decimals=0)),
			int(np.round((sAP/items*100),decimals=0)),
		)
	)

	print()

def generate_keras_prediction_batch(x_batch,keras_model,data_generator,**kwargs):
	''' Takes a batch of numpy images, runs detection inference
	with given keras model and transforms the results into a list of 
	predictions. The prediction list is sorted in decreasing order
	along the confidence score.

	Args:
		x_batch: [height,width,3] image data in rgb format
		keras_model: instance of keras model

	Returns:
		batch_predictions: [num_predictions,5]
	'''

	predictions_list = keras_model.predict(x_batch, batch_size=x_batch.shape[0], verbose=0)
	batch_predictions = np.empty((0,))
	for item_id in range(x_batch.shape[0]):
		item_feature_map_predictions = []
		for output_level in predictions_list:
			item_feature_map_predictions.append(output_level[item_id])
		linear_predictions_vector,_ = data_generator.invert_fit_to_model_outputs(item_feature_map_predictions)
		linear_predictions_vector_sorted = data_generator.decode_predictions_sorted(linear_predictions_vector,**kwargs)		
		if len(batch_predictions) == 0:
			batch_predictions = np.expand_dims(linear_predictions_vector_sorted,axis=0)
		else:
			batch_predictions = np.append(batch_predictions,np.expand_dims(linear_predictions_vector_sorted,axis=0),axis=0)	
	return batch_predictions	

def evaluate_keras_model(**kwargs):	
	image_size = DEFAULT_IMG_SIZE
	grid_sizes = [27,13,7,3]
	model_dir = kwargs['model_dir']
	model = generate_model(model_dir)
	image_size = model.input_shape[1:]
	grid_sizes = []
	for output_tensor in model.outputs:
		grid_sizes.append(output_tensor.shape[1].value)

	if kwargs['prediction_mode']:
		data_generator = data_sequence.Prediction_Data_Sequence(
			image_path=kwargs['markup_path'],
			image_size=image_size,
			grid_sizes=grid_sizes,
			**kwargs)
	else:
		data_generator = data_sequence.Simple_Bounding_Box_Sequence(
			image_size=image_size,
			grid_sizes=grid_sizes,
			**kwargs)

	print(data_generator.grid_sizes)
	if len(data_generator) == 0:
		raise Exception('no annotated data provided.')

	for ckpt in kwargs['weights_ckpt_file']:
		ckpt_path_parts = ckpt.split('/')
		ckpt_name,_ = os.path.splitext(ckpt_path_parts[-1])
		model_name = ckpt_path_parts[-2]
		print('use model at checkpoint {}'.format(ckpt_name))
		model.load_weights(ckpt)		
		
		found = 0
		total = 0
		conf = 0.0
		iou = 0.0
		negs = 0
		negs_inv = 0
		neg_conf = 0.0
		neg_conf_inv = 0.0
		cummulative_ap = 0.0
		cummulative_d_ap = 0.0
		cummulative_s_ap = 0.0
		cummulative_coverage = 0.0
		items = 0
		for h in range(len(data_generator)):
			print('processing batch {}/{}...'.format(h+1,len(data_generator)))
			x_batch,y_batch = data_generator[h]
			batch_predictions = generate_keras_prediction_batch(x_batch,model,data_generator,**kwargs)
			assert batch_predictions.shape[0] == len(x_batch),(
				'batch size error: predictions {} do not fit to size of batch {}'.format(batch_predictions.shape[0],len(x_batch)))
			#print(batch_predictions.shape)#(8, 5736, 5)
			#print(x_batch.shape)#(8, 460, 460, 3)
			for i in range(batch_predictions.shape[0]):
				predictions = batch_predictions[i]
				pred_image_np = classification_auxiliary.invert_tf_preprocessor(x_batch[i])
				if kwargs['prediction_mode']:
					detection_index_list,positive_detections = detect(predictions,**kwargs)
					
					detection_output_root = os.path.join(DEFAULT_OUTPUT_SUBDIR,DETECT_DATA_SUBDIR_NAME,model_name,ckpt_name)
					img_subdir = os.path.join(
						detection_output_root,
						y_batch[i][0])
					os.makedirs(img_subdir,exist_ok=True)

					img_file_name = '{}__detections_{}of{}__iou_{:.2f}__conf_{:.2f}{}'.format(
						y_batch[i][1],
						positive_detections.shape[0],
						kwargs['max_number_of_detections'],
						kwargs['iou_overlap_threshold'],
						kwargs['confidence_threshold'],
						y_batch[i][2], # file extension
						)
					img_output_path = os.path.join(
						img_subdir,
						img_file_name
						)

					plot_preds_to_img(pred_image_np,positive_detections,color='blue',thickness=12,plot_conf_label=True,**kwargs)
					plot_preds_to_img(pred_image_np,positive_detections,color='yellow',thickness=6,plot_conf_label=True,**kwargs)
					save_img(img_output_path,pred_image_np)
				else:
					detection_output_root = os.path.join(DEFAULT_OUTPUT_SUBDIR,EVAL_DATA_SUBDIR_NAME,model_name,ckpt_name)
					ground_truth = y_batch[i]
					
					ap,distinct_ap,steady_ap,coverage = average_precision(predictions,ground_truth,**kwargs)
					cummulative_ap += ap
					cummulative_d_ap += distinct_ap
					cummulative_s_ap += steady_ap
					cummulative_coverage += coverage
					items += 1

					(found_objects,
						confidence_sum,
						iou_sum,
						neg_conf_sum,
						neg_sum,
						neg_conf_inv_sum,
						neg_inv_sum) = evaluate(predictions,ground_truth,**kwargs)
					found += found_objects.shape[0]
					total += ground_truth.shape[0]
					conf += confidence_sum
					iou += iou_sum
					negs += neg_sum
					neg_conf += neg_conf_sum
					negs_inv += neg_inv_sum
					neg_conf_inv += neg_conf_inv_sum
				
					if kwargs['output_flag']:
						detection_index_list,_ = detect(predictions,**kwargs)
						os.makedirs(detection_output_root,exist_ok=True)

						plot_to_img(pred_image_np,detection_index_list,predictions,'red',12,True) # for each gt the most confident box with iou > 0.6
						plot_to_img(pred_image_np,detection_index_list,predictions,'white',6,True) # for each gt the most confident box with iou > 0.6
						save_img(os.path.join(detection_output_root,'detections_{}.jpg'.format(h*data_generator.batch_size+i)),pred_image_np)

			if not kwargs['prediction_mode']:
				print_result_state(found,total,cummulative_ap,cummulative_d_ap,cummulative_s_ap,items,cummulative_coverage,kwargs['pos_iou_assign_threshold'])

		if not kwargs['prediction_mode']:
			print()
			output_eval_result(ckpt,cummulative_ap,cummulative_d_ap,cummulative_s_ap,items,found,total,conf,iou,negs,neg_conf_sum,neg_conf_inv,negs_inv,coverage=cummulative_coverage,**kwargs)
		if kwargs['prediction_mode'] or kwargs['output_flag']:
			print('plotted detections are saved at: {}'.format(os.path.join(DEFAULT_OUTPUT_SUBDIR,DETECT_DATA_SUBDIR_NAME,model_name)))
			print()

def generate_tf_prediction_batch(x_batch,detection_graph):
	''' Takes a batch of numpy images, runs detection inference
	with given tensorflow graph and transforms the results into a 
	list of predictions. The prediction list is sorted in decreasing 
	order along the confidence score.

	Args:
		x_batch: [height,width,3] image data in rgb format
		detection_graph: instance of tensorflow object detection
			api model

	Returns:
		batch_predictions: [num_predictions,5]
	'''
	batch_predictions = np.empty((0,))
	for image_np in x_batch:
		output_dict = generateAnnotationFromModelsPredictions.run_inference_for_single_image(image_np,detection_graph)
		predictions = np.empty((output_dict['num_detections'],5))
		for j in range(output_dict['num_detections']):
			box_j = data_sequence.Keras_Detection_Sequence.convert_min_max_to_norm(output_dict['detection_boxes'][j])
			predictions[j,:4] = box_j
			predictions[j,-1] = output_dict['detection_scores'][j]
		if len(batch_predictions) == 0:
			batch_predictions = np.expand_dims(predictions,axis=0)
		else:
			batch_predictions = np.append(batch_predictions,np.expand_dims(predictions,axis=0),axis=0)
	return batch_predictions

def evaluate_frozen_tf_model(**kwargs):
	image_size = DEFAULT_IMG_SIZE
	grid_sizes = [0,0]

	if kwargs['prediction_mode']:
		data_generator = data_sequence.Prediction_Data_Sequence(
			image_path=kwargs['markup_path'],
			image_size=image_size,
			grid_sizes=grid_sizes,
			**kwargs)
	else:
		data_generator = data_sequence.Simple_Bounding_Box_Sequence(
			image_size=image_size,
			grid_sizes=grid_sizes,
			**kwargs)

	for ckpt in kwargs['weights_ckpt_file']:
		detection_graph = generateAnnotationFromModelsPredictions.load_tf_graph(ckpt)
		print('evaluate model at checkpoint {}'.format(ckpt))
		ckpt_dir = ckpt.split('/')[:-1]
		ckpt_name = ckpt_dir[-1]
		model_name = ckpt_dir[-2]

		found = 0
		total = 0
		conf = 0.0
		iou = 0.0
		negs = 0
		negs_inv = 0
		neg_conf = 0.0
		neg_conf_inv = 0.0
		cummulative_ap = 0.0
		cummulative_d_ap = 0.0
		cummulative_s_ap = 0.0
		cummulative_coverage = 0.0
		items = 0
		for h in range(len(data_generator)):
			print('processing batch {}/{}...'.format(h+1,len(data_generator)))
			x_batch,y_batch = data_generator[h]
			batch_predictions = generate_tf_prediction_batch(x_batch,detection_graph)
			assert batch_predictions.shape[0] == len(x_batch),(
				'batch size error: predictions {} do not fit to size of batch {}'.format(batch_predictions.shape[0],len(x_batch)))
			for i in range(batch_predictions.shape[0]):
				predictions = batch_predictions[i]
				pred_image_np = x_batch[i]
				if kwargs['prediction_mode']:
					detection_index_list,positive_detections = detect(predictions,**kwargs)
					
					detection_output_root = os.path.join(DEFAULT_OUTPUT_SUBDIR,DETECT_DATA_SUBDIR_NAME,model_name,ckpt_name)
					img_subdir = os.path.join(
						detection_output_root,
						y_batch[i][0])
					os.makedirs(img_subdir,exist_ok=True)
					
					img_file_name = '{}__detections_{}of{}__iou_{:.2f}__conf_{:.2f}{}'.format(
						y_batch[i][1],
						positive_detections.shape[0],
						kwargs['max_number_of_detections'],
						kwargs['iou_overlap_threshold'],
						kwargs['confidence_threshold'],
						y_batch[i][2], # file extension
						)
					img_output_path = os.path.join(
						img_subdir,
						img_file_name
						)

					plot_preds_to_img(pred_image_np,positive_detections,color='blue',thickness=12,plot_conf_label=True,**kwargs)
					plot_preds_to_img(pred_image_np,positive_detections,color='yellow',thickness=6,plot_conf_label=True,**kwargs)
					save_img(img_output_path,pred_image_np)
				else:
					detection_output_root = os.path.join(DEFAULT_OUTPUT_SUBDIR,EVAL_DATA_SUBDIR_NAME,model_name,ckpt_name)
					ground_truth = y_batch[i]
					
					(ap,
						distinct_ap,
						steady_ap,
						coverage) = average_precision(predictions,ground_truth,**kwargs)
					cummulative_ap += ap
					cummulative_d_ap += distinct_ap
					cummulative_s_ap += steady_ap
					cummulative_coverage += coverage
					items += 1
					
					(found_objects,
						confidence_sum,
						iou_sum,
						neg_conf_sum,
						neg_sum,
						neg_conf_inv_sum,
						neg_inv_sum) = evaluate(predictions,ground_truth,**kwargs)
					found += found_objects.shape[0]
					total += ground_truth.shape[0]
					conf += confidence_sum
					iou += iou_sum
					negs += neg_sum
					neg_conf += neg_conf_sum
					negs_inv += neg_inv_sum
					neg_conf_inv += neg_conf_inv_sum

					if kwargs['output_flag']:
						detection_index_list,_ = detect(predictions,**kwargs)
						os.makedirs(detection_output_root,exist_ok=True)

						plot_to_img(pred_image_np,detection_index_list,predictions,'red',12,True)
						plot_to_img(pred_image_np,detection_index_list,predictions,'white',6,True)
						save_img(os.path.join(detection_output_root,'detections_{}.jpg'.format(h*data_generator.batch_size+i)),pred_image_np)
			if not kwargs['prediction_mode']:
				print_result_state(found,total,cummulative_ap,cummulative_d_ap,cummulative_s_ap,items,cummulative_coverage,kwargs['pos_iou_assign_threshold'])
		if not kwargs['prediction_mode']:
			print()
			output_eval_result(ckpt,cummulative_ap,cummulative_d_ap,cummulative_s_ap,items,found,total,conf,iou,negs,neg_conf_sum,neg_conf_inv,negs_inv,coverage=cummulative_coverage,**kwargs)
		else:
			print('plotted detections are saved at: {}'.format(os.path.join(DEFAULT_OUTPUT_SUBDIR,DETECT_DATA_SUBDIR_NAME,model_name)))
			print()

def main(**kwargs):
	if kwargs['keras_preprocessing_mode']:
		evaluate_keras_model(**kwargs)
	else:
		evaluate_frozen_tf_model(**kwargs)

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