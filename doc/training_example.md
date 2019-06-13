# Run a Training Process for a Keras Object Detector

This manual assumes that a bunch of images have been annotated with the [labelImg](https://github.com/tzutalin/labelImg) tool. Further, the annotations are partitioned into two distinct datasets (*train* and *test* data) and the xml markup files of each dataset are ensembled in a separate subdirectory under a common root directory (the training process directory).
The partitioning of the data can be done with the `utils/extractAnnotations.py` script (see <a href='doc/preparing_inputs.md'>preparing inputs</a>). The script automatically creates a directory `{label_of_training_process}` at the `data/trainings` path where the datasets of the training process are organized.

**Object detectors that are implemented with Keras only require _valid xml markup files_ as input.**

It is also assumed, that a valid pipeline configuration file exists (e.g., in the `config` directory). For the configuration it is important that the *path to the xml markup files* is specified in the **detection_config** dictionary under the key **data_path** (e.g., `data_path: data/trainings/{label_of_training_process}/annotations`).
A sample configuration can be found at <a href='config/default_pipeline.yaml/'>`config/default_pipeline.yaml`</a>.

The training process can be started with the following command:
```bash
# From Graffiti_Detection/
python train_inceptionV3_detector.py \
  --path_to_config_file=config/default_pipeline.yaml
```

The script creates and manages the *model directory*. When the process is finished, the model directory contains all saved checkpoints, the model configuration file, the pipeline file and some log data for the training process (see <a href='doc/file_structure.md'>file organisation</a>). The basic structure of the model directory looks as follows:

```
+models
  +keras_models
    +{model_architecture}
      +{model_dir}
        +logs
        -weights_init_classification_00.hdf5
        -pipeline.yaml
        -model_config.yaml
        -{weight_checkpoint_1}.hdf5
        -...
        -{history_1}.yaml
        -...
```
