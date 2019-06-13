# Model Configuration

## General Approach

The basic approach to obtain an object detector consists of the following steps:
1. Pick a Keras **classification model**.
2. Further **fine tune** the classification model on a new set of images [this step is optional].
3. Use a classification model to **set up and train a _detection pipeline_**.

[Keras Applications](https://keras.io/applications/) provides a set of
pre-trained classification models that can be used to build a detection pipeline.

## Configuration File

The configuration file manifests the setup of your pipeline. The file has to
follow the [yaml standard](https://yaml.org/).

Sample configurations are provided in the <a href='config/'>`config`</a> directory.

### Base Fields
|**Field name**|**Description**|
|:---|:---|
|`base`|Name of the Keras classification model that is used as backbone cnn (currently only _InceptionV3_ is supported).|
|`base_output_layer`|name of the cnn's bottleneck layer; the detection head will be attached to this layer|
|`input_shape`|python tuple defines the input resolution that is accepted by the backbone cnn|
|`feature_map_layers`|list of the layer names that serve as base layers for building the feature map|
|`base_model_dir`|directory to a classification model that should be used as a fallback to generate a detection model if classification config fields are not provided|
|`base_model_ckpt`|checkpoint name of the fallback classification|
|`classification_config`|(see [Classification Config Fields](#Classification-Config-Fields))|
|`detection_config`|(see [Detection Config Fields](#Detection-Config-Fields))|

#### Detection Config Fields
|**Field name**|**Description**|
|:---|:---|
|`subdir_name`|prefix of the model subdir|
|`aspect_ratios`|list of different aspect ratios for default anchor boxes|
|`min_scale`|minimum scale of default anchor boxes|
|`max_scale`|maximum scale of default anchor boxes|
|`max_anchor_limit`|width and height of anchor boxes are clipped to this value if dimensions exceed the limit (between 0.0 and 1.0)|
|`feature_map_filter_sizes`|[optional/default:256] depth of the linear feature map layers|
|`default_kernel_size`|[optional/default:3] indicates the default kernel size that is used to generate the linear feature maps from the base layers and default kernel size of the detection head convolutions|
|`min_kernel_size`|[optional/default:3] indicates a lower bound to which the feature map layers can be reduced in size|
|`feature_pyramid_network`|[optional/default:false] boolean indicates whether architecture should apply a feature pyramid network|
|`fpn_filter_sizes`|[optional/default:256] depth of the feature pyramid network layers|
|`parameter_sharing`|[optional/default:false] boolean whether to apply parameter sharing between the different feature pyramid layers|
|`spatial_drop_out_rate`|[optional/default:0.0] if specified a drop out layer is included between the feature pyramid and the detection head with the given drop out rate (recommended to prevent overfitting during training)|
|`weight_checkpoint_file`|name of the detection model checkpoint that should be used to initialize the model weights for training|
|`enabled`|[optional/default:true] if true training process is enabled|
|`data_path`|root path to a directory with annotations files, the directory is processed recursively to generate the full train/test datasets|
|`train_test_split`|[optional/default:false] if true the data is split into distinct sets for training and test/validation loops|
|`test_dir_pattern`|[optional/default:test] name of the directory that contains test data|
|`pos_iou_assign_threshold`|[optional/default:0.5] IoU threshold for the assignment of ground truth boxes to default anchor boxes, (and thus to feature map locations); the default anchor box gets assigned a positive label|
|`neg_iou_assign_threshold`|[optional/default:0.4] IoU threshold for the assignment of negative labels to default anchor boxes|
|`similarity_func`|[optional/default:jaccard_similarity] method for IoU calculation|
|`batch_size`|[optional/default:4] batch size for the training process|
|`loss`|list of names of the loss function that should be used for the training process (a separate model is trained for loss function)|
|`sigma`|[optional/default:2.0] parameter to scale the focal loss modulation factor|
|`alpha`|list of parameters to weights between localization and classification loss (a separate model is trained for each weight)|
|`trainings`|Trainings Fields (dictionary) for configuring the training stages|

#### Trainings Fields
|**Field name**|**Description**|
|:---|:---|
|`epochs`|[optional/default:30]|
|`enabled`|[optional/default:true] if true training stage is enabled|
|`freeze_to_bottleneck`|[optional/default:false] freeze model to bottleneck layer if true, else freeze model to specified fine_tuning_layer for this trainig stage|
|`fine_tuning_layer`|\[optional/default:base_output_layer (see [Base Fields](#Base-Fields))\] name of the layer that should serve as the freezing point for the weight updates|
|`optimizer`|dictionary where key is the name of an optimizer from *keras.optimizers* library which should be applied to the training stage and the value is a dict of kwargs settings that are passed to the optimizer's constructor (see https://keras.io/optimizers/)|
|`callbacks`|dictionary where key is the name of a callback from *keras.callbacks* library which should be applied to the training stage and the value is a dict of kwargs settings that are passed to the callback's constructor (see https://keras.io/callbacks/)|

### Classification Config Fields
|**Field name**|**Description**|
|:---|:---|
|`head_layers`|Head Layer Fields (dictionary), defines the structure of the classification head|
|`weight_checkpoint_file`|name of the classification model checkpoint that should be used to initialize the model weights for training|
|`enabled`|[optional/default:true] if true training process is enabled|
|`validation_split`|[optional/default:0.0] fraction of training data that should be reserved for validation at the end of each epoch|
|`data`|dictionary {*train_items_path*:path_to_train_images,*val_items_path*:path_to_val_images} containing paths to train and validation datasets |
|`batch_size`|[optional/default:4]|
|`loss`|[optional/default:categorical_crossentropy] name of a keras loss function that should be applied for training (see https://keras.io/losses/)|
|`metrics`|[optional/default:\[accuracy\]] list of keras metric names that should be computed during training (see https://keras.io/metrics/)|
|`classification_training`|Trainings Fields (dictionary) for configuring the training stages|

#### Head Layer Fields
|**Field name**|**Description**|
|:---|:---|
|`conv2d`|list of *keras.layers.Conv2D* encoded as python tuples in the following order (filters, kernel_size, strides, padding, activation, alpha) where the alpha param is applied to *keras.layers.LeakyReLU* layer if activation is 'relu'|
|`averagePooling2D`|[optional/default:0] either 1 or 0 or boolean value, if true one layer of this kind is attached to the convolutional layers|
|`maxPooling2D`|[optional/default:0] either 1 or 0 or boolean value, if true one layer of this kind is attached to the convolutional layers|
|`dense`|list of *keras.layers.Dense* encoded as python tuples in the following order (units, activation, alpha)where the alpha param is applied to *keras.layers.LeakyReLU* layer if activation is 'relu'|
|`dropout`|list of dropout values that are added between and after the dense layers|
