# Preparing Inputs for Object Detection with Keras and Tensorflow

Object detection with Keras and tensorflow object detection api requires a set of bounding box annotations that are generated with the [labelImg](https://github.com/tzutalin/labelImg) tool.
**Important:** Make sure that the **_path_ field** in the xml files contains *valid* paths to images on disk.

## Data of a Training Process

The annotation xml markup files need to be organized before you can use them to train a model.
For each *training process*, the data is **split** into two distinct datasets (**training** and **test/validation**). You can do this by simply organizing the data into separate directories for *test* and *train* data in a root directory that indicates the training process label.

The <a href='doc/file_structure.md'>file organisation / directory structure</a> of this framework maintains the path `data/trainings` where the data for different training processes are organized. Thus, the training process can be carried out multiple times (with the same data) to obtain different detection models. The basic file structure looks as follows:

```
+data
  +trainings
    +{label_of_training_process_1}
      +train
      +test
    +{label_of_training_process_2}
    +...
```

The xml markup files that should be used to train a model are located (directly or in a subdirectory) under `data/trainings/{label_of_training_process_1}/train`. The validation data for *label_of_training_process_1* is placed in the `test` folder.

## Organizing a Training Process automatically

You can use the script `utils/extractAnnotations.py` to prepare a set of xml markup files for the training process. The script automatically creates the directories for the training process in the `data/trainings` path, makes the data split and copies the xml markup files into the `test` and `train` subdirectories.

As input, the script only requires a path to a root directory where all markup files are located (either directly or in subdirectories) that should be used for the training process. The images can but need not to be located along with the markup files.

Some sample data is located in the <a href='data/example_data/images_and_annotations'>`data/example_data/images_and_annotations`</a> and <a href='data/example_data/annotations_only'>`data/example_data/annotations_only`</a> directories. The data is organized into different subdirectories (e.g., each represents a different data source).

The following commands use these directories and create two different training processes with labels *example_training_1* and *example_training_2*. The data is automatically split into distinct sets for training and test/validation according to the provided value for the  `--test_fraction` parameter:

```bash
# From Graffiti_Detection/
python utils/extractAnnotations.py \
  --markup_root=data/example_data/images_and_annotations/ \
  --training_name=example_training_1 \
  --test_fraction=0.3
```

```bash
# From Graffiti_Detection/
python utils/extractAnnotations.py \
  --markup_root=data/example_data/annotations_only/ \
  --training_name=example_training_2 \
  --test_fraction=0.5 \
  --image_root_dir=data/images/train_detection_dataset
```

After this command has finished, a training directory for each training process is created at `data/trainings/example_training_1` and `data/trainings/example_training_2`. The xml markup files are split into test and train sets. More infos about the resulting file structure can be found <a href='doc/file_structure.md'>here</a>.

```
+data
  +trainings
    +{example_training_1}
      +annotations
        +train
          +{subdir_1}
            -{markup_1}.xml
            -...
          +{subdir_2}
          +...
        +test
          +{subdir_1}
            -{markup_2}.xml
            -...
          +{subdir_2}
          +...
```

## Adding TFRecords

If the training data should be used with the tensorflow object detection api, an additional step is required to generate the corresponding TFRecords for the training data. The following command takes the train and test datasets (as organized in the `annotations` subdirectory) and turns them into proper TFRecords that are stored in a `TFRecords` subdirectory.

```bash
# From Graffiti_Detection/
python utils/generateTfRecord.py \
  --path_to_training_directory=data/trainings/example_training_1 \
  --test_item_pattern=test
```

This command will add the following structure to the `data/trainings/example_training_1` directory:

```
+data
  +trainings
    +{example_training_1}
      +annotations
      +TFRecords
        -data_info.yaml
        -label_map.pbtxt
        -test_dataset.record-00000-of-?????
        -...
        -train_dataset.record-00000-of-?????
        -...
```
