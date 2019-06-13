# Using Keras models for object detection

The saved checkpoints in the `{model_dir}` represent the different models. You can use these models to
1. **evaluate** the detection capabilities for a set of xml markup files or
2. apply the model to a set of images and **detect objects**.

These use cases are covered by the script <a href='eval_model.py'>`eval_model.py`</a> (please look directly at the file for more detailed information about all available parameters).
In addition a *jupyter notebook* is provided at <a href='graffiti_detection_with_keras_example.ipynb'>`graffiti_detection_with_keras_example.ipynb`</a> that illustrates a step-by-step approach how object detection in images actually works with Keras models.

## Evaluating Keras models

Say you have <a href='doc/training_example.md'>run and finished a training process</a> and want to evaluate a model from the automatically created `{model_dir}` for a set of xml markup files. First you need to specify the parameters:

```bash
model_dir=models/keras_models/Lfm_SeparatedParams_drop_0.5/
test_data=data/trainings/graffiti_ba_training/annotations/test
iou=0.5
conf=0.2
```

Then, you can start the evaluation of the models with the following command:

```bash
# From Graffiti_Detection/
python eval_model.py \
  --model_dir=${model_dir} \
  --input_data_dir=${test_data} \
  --iou_boundary=${iou} \
  --confidence_threshold=${conf}
```

The script list all evailable checkpoints/models that can be found in the `{model_dir}`. You'll have to select at least one that should be evaluated.
It is important, that the `input_data_dir` specifies a path to a directory that contains _xml markup files_.

The results are printed to an `eval.txt` file that is located in the model directory:
```
+models
  +keras_models
    +{model_dir}
      -eval.txt
      -...
```

In addition, the `--output_flag` parameter can be passed to the script. When the parameter is present, the script will also plot the detection results for the images to disk. The detections can be found in the directory `data/evaluation_output`:

```
+data
  +evaluation_output
    +{model_name}
      +{ckpt_name}
        -detections_1.jpg
        -detections_2.jpg
        -...
```

## Object Detection with Keras models

When you want to use a Keras model for object detection, you simply need to pass the `--detection_only` parameter to the script. In that case, the directory which is passed to the  `--input_data_dir` parameter should contain only images (or only xml markup files):

```bash
model_dir=models/keras_models/Lfm_SeparatedParams_drop_0.5/
test_data=data/images/detection_test/
iou=0.4
conf=0.2
```

The object detection is executed with the following command:

```bash
# From Graffiti_Detection/
python eval_model.py \
  --model_dir=${model_dir} \
  --input_data_dir=${test_data} \
  --iou_boundary=${iou} \
  --confidence_threshold=${conf} \
  --detection_only
```

The detections are plotted to disk at `data/detection_output`:
```
+data
  +detection_output
    +{model_name}
      +{ckpt_name}
        +{test_data}_{timestamp}
          -{img_name}__detections_{n}of{N}__iou_{iou}__conf_{conf}.jpg
          -{img_name}__detections_{n}of{N}__iou_{iou}__conf_{conf}.jpg
          -...
```

<p align="center">
  <img src="doc/img/lfm_detections.jpg" width=450 height=450>
</p>
