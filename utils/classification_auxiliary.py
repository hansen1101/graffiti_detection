import keras
import keras.applications.imagenet_utils
import tensorflow as tf
import numpy as np

def tf_preprocessor(image_np):
  ''' Function simply applies keras applications imagenet preprocessing in tf mode to an image.
  It also checks for the required invariant that input shape equals output shape such that this
  function meets the requirements of the ImageDataGenerator's preprocessing_function parameter.
  When this function is passed as preprocessing_function parameter to an ImageDataGenerator
  the function is applied to all input images after each image is resized and augmented.

  # Args
    numpy tensor representing one image (Numpy tensor with rank 3)
  # Returns
    numpy tensor with the same shape as input
  '''
  #print('preprocess item ',np.amax(image_np),np.amin(image_np))
  #image_np /= 127.5
  #image_np -= 1.
  #return image_np
  tmp = np.array(
    keras.applications.imagenet_utils.preprocess_input(image_np,mode='tf') # since preprocess_input returns element of type tuple containing the image array as only element
    )
  assert np.array_equal(tmp.shape,image_np.shape)
  return tmp

def invert_tf_preprocessor(image_np):
  tmp = np.copy(image_np)
  tmp += 1.
  tmp *= 127.5
  return np.around(tmp).astype(int)

def freezeModelToBlock(model,layer_names,start_training_after_layers=False):
  '''
  Takes a list of layer names representing a level in the model up to which the model's layer
  should be frozen for the training/fittment.
  # Args:
    model pointer: the model to which the freezing should be applied
    layer_names list of string: names of the layers which represent the starting point
      for setting the trainable property to true
    start_training_after_layers boolean whether to include the provided layer names in
      the frozen part of the model or not
  # Usage:
    freezeModelToConvBlock(model,['conv2d_%d'%i for i in [86,88,89,92,93,94]])
    freezes from layers in ['conv2d_86', 'conv2d_88', 'conv2d_89', 'conv2d_92', 'conv2d_93', 'conv2d_94']
  '''
  input_stack = [] # list for tensors
  for i, layer in enumerate(model.layers):
    is_trainable = False # state to which the layer will be set
    append_successors = False # helper flag if training should start after the given layer list
    if layer.name in layer_names:
      # case we are at a starting layer
      if not start_training_after_layers:
        is_trainable=True
      else:
        append_successors = True
    elif input_stack:
      # input_stack contains tensors
      if isinstance(layer.input,tf.Tensor):
        # layer gets input from a single tensor:
        if layer.input in input_stack:
          # set layer to be trainable since tensor's output generating layer is also trainable
          is_trainable=True
          # reslice input_stack
          position = input_stack.index(layer.input)
          #input_stack = input_stack[:position]+input_stack[position+1:]
      elif isinstance(layer.input,list):
        # layer gets input from a multiple tensors:
        if [candidate for candidate in input_stack if candidate in layer.input]:
          # at least one of the layers input tensors is included in the input_stack
          is_trainable=True
          # reslice input_stack
          for tensor in layer.input:
            try:
              position = input_stack.index(tensor)
              #input_stack = input_stack[:position]+input_stack[position+1:]
            except:
              pass
    layer.trainable=is_trainable # set the layer state
    
    # if the layer is set to be trainable, add all output tensors to the input_stack 
    # such that the layers depending on the tensors eventually will become trainable.
    if is_trainable or append_successors:
      j = 0
      try:
        pass
        while True:
          output_tensor = layer.get_output_at(j)
          if output_tensor is not None:
            input_stack.append(output_tensor)
            j += 1
          else:
            break
      except:
        pass
        '''    
        if isinstance(layer.output,list):
          input_stack += layer.output
        else:
          input_stack.append(layer.output)
        '''