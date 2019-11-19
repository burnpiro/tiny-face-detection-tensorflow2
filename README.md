# Tiny Face Detection with TensorFlow 2.0

![alt text][image]

### Quick start

- Install tensorflow and other `requirements.txt`
- Get [DataSet](#dataset)
- run `python train.py` (takes a while, depends on your machine)
- run `python detect.py --image my_image.jpg`

### Important files

- [`./data/data_generator.py`](#data_generatorpy) - generates train/val data from WIDER FACE
- [`./model/model.py`](#modelpy) - generates TF model
- [`./model/loss.py`](#losspy) - definition of the **loss function** for training
- [`./model/validation.py`](#validationpy) - definition of the validation for training
- [`./config.py`](#configpy) - stores network/training/validation config for network
- [`./detect.py`](#detectpy) - runs model against given image and generates output image
- `./draw_boxes` - helper function for `./detect.py`, draws boxes on cv2 img
- `./print_model.py` - prints current model structure
- [`./train.py`](#trainpy) - starts training our model and create weights base on training results and validation function

### Dataset

We want to use WIDER FACE dataset. It contain over 32k images with almost 400k faces and is publicly available on
[http://shuoyang1213.me/WIDERFACE/](http://shuoyang1213.me/WIDERFACE/)

- [Training Data GDrive](https://drive.google.com/file/d/0B6eKvaijfFUDQUUwd21EckhUbWs/view?usp=sharing)
- [Val Data GDrive](https://drive.google.com/file/d/0B6eKvaijfFUDd3dIRmpvSk8tLUk/view?usp=sharing)
- [Test Data GDrive](https://drive.google.com/file/d/0B6eKvaijfFUDbW4tdGpaYjgzZkU/view?usp=sharing)
- [Annotations](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip)

Please put all the data into `./data` folder.

Data structure is described in `./data/wider_face_split/readme.txt`. We only need to use boxes annotations but there is more data available if someone wants to use it.

### Files

#### data_generator.py

@config_path - path to `data/wider_face_split/wider_face_train_bbx_gt.txt` file (defined in `cfg.TRAIN.ANNOTATION_PATH`)
@file_path - path to folder with images (defined in `cgf.TRAIN.DATA_PATH`)

- `__init__(file_path, config_path, debug=False)` loops over all images in txt file (base on `config_path`) and stores them inside generator to be retrieved by `__getitem__`
- `__len__()` unsurprisingly returns length of our data (exactly number of batches `data/batch_size)
- `__getitem__(idx)` - returns data for given `idx`, data returned as `Array<imagePath>, Array<h, w, yc, xc, class>`

#### model.py

- `create_model(trainable=False)` - creates model base on definition, if you want model to be fully trainable (not only output layers) then set `trainable` to be `True`

#### loss.py

- `loss(y_true, y_pred)` - returns value of **loss function** for current prediction (`y_true` is a box from dataset, `y_pred` is a output from NN)
- `get_box_highest_percentage(arr)` - helper function for `loss` to get best box match

#### validation.py

- `on_epoch_end(self, epoch, logs)` - calculates `IoU` and `mse` for validation set
- `get_box_highets_percentage(self, mask)` - helper function, you can ignore it

#### config.py

Just a config, there is couple of important things in it:
- `ALPHA` - mobilenet's "alpha" size, higher value means more complex network (slower, more precise)
- `GRID_SIZE` - output grid size, **7** is a good value for low ALPHA but you might want to set it to higher value for larger ALPHAs and add UpSample layer to model.py
- `INPUT_SIZE` - value should be adjusted base on initial network used (**224** for MobileNetV2, but check input size if you changing model)
 
Inside `TRAIN` prefix there is couple training hyperparameters you can adjust for training

#### detect.py

You have to first train model to get at least one `model-0.xx.h5` weights file

Usage:
```bash
# basic usage
python detect.py --image path_to_my_image.jpg

# use different trained weights and output path
python detect.py --image path_to_my_image.jpg --weights model-0.64.h5 --output output_path.jpg
```

#### train.py

There is no parameters for it but you might want to read that file. It's running base on `config.py` and other files already described. If you want to train your model from specific point then uncomment `IF TRAINABLE` and add weights file.

After running training script will generate `./logs/fit/**` files. You can use **Tensorboard** for visualise training

```bash
tensorboard --logdir logs/fit
```


[image]: ./example_output.jpg "Sample Output"