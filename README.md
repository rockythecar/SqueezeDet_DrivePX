## _SqueezeDet:_ Unified, Small, Low Power Fully Convolutional Neural Networks for Real-Time Object Detection for Autonomous Driving

This repository contains a initial tensorflow implementation of SqueezeDet, a convolutional neural network based object detector described in their paper: https://arxiv.org/abs/1612.01051.

This version was amended for the drivePX

## Version:

- demo_v1_crop.py

  ```Shell
  Choose the center part of the image, no resize. Not for drivePX.
  ```

- demo_v3_PIL.py

  ```Shell
  Resize. DrivePX version.
  ```


## Installation:

The following instructions are written for Linux-based distros.

### DrivePX

- Clone the SqueezeDet repository:

  ```Shell
  git clone git@github.com:honda-research-institute/SqueezeDet_DrivePX.git
  ```
  Let's call the top level directory of SqueezeDet `$SQDT_ROOT`. 
  
- Load nvidia docker [tensorflow]:

- Use pip to install required Python packages:
  
  1. (Optional) Install pip if the DrivePx doesn't have it:
    
    ```Shell
    sudo apt-get update
    sudo apt-get install python-pip
    ```
    
  2. Edit requirements file, remove openCV and tensorflow    
    ```Shell
    easydict==1.6
    joblib==0.10.3
    numpy==1.12.0
    Pillow==4.0.0
    ```
   
  3. Install package for python    
    ```Shell
    pip install -r requirements.txt
    ```

- Download SqueezeDet model parameters from [here](https://www.dropbox.com/s/a6t3er8f03gdl4z/model_checkpoints.tgz?dl=0), untar it, and put it under `$SQDT_ROOT/data/` If you are using command line, type:

  ```Shell
  cd $SQDT_ROOT/data/
  wget https://www.dropbox.com/s/a6t3er8f03gdl4z/model_checkpoints.tgz
  tar -xzvf model_checkpoints.tgz
  rm model_checkpoints.tgz
  ```


- Now we can run the program. Make sure you download photos into the "input_path"
  ```Shell
  python3 ./src/demo_v3_PIL.py  --input_path="./../one_half_x/object-detection-crowdai/*.jpg"  --out_dir="result/"
  ```

### NonDrivePX

- Clone the SqueezeDet from DIFFERENT repository:

  ```Shell
  git clone git@github.com:BichenWuUCB/squeezeDet.git
  ```
  Let's call the top level directory of SqueezeDet `$SQDT_ROOT`. 
  
For example, the local PC with ubuntu

- (Optional) Setup your own virtual environment.

  1. The following assumes `python` is the Python2.7 executable. Navigate to your user home directory, and create the virtual environment there.
  
    ```Shell
    cd ~
    virtualenv env --python=python
    ```
    
  2. Launch the virtual environment.
  
    ```Shell
    source env/bin/activate
    ```
- Use pip to install required Python packages:
  
  1. (Optional) Install pip if the your machine doesn't have it:
    
    ```Shell
    sudo apt-get update
    sudo apt-get install python-pip
    ```  
   
  2. Install package for python    
    ```Shell
    pip install -r requirements.txt
    ```

- Download SqueezeDet model parameters from [here](https://www.dropbox.com/s/a6t3er8f03gdl4z/model_checkpoints.tgz?dl=0), untar it, and put it under `$SQDT_ROOT/data/` If you are using command line, type:

  ```Shell
  cd $SQDT_ROOT/data/
  wget https://www.dropbox.com/s/a6t3er8f03gdl4z/model_checkpoints.tgz
  tar -xzvf model_checkpoints.tgz
  rm model_checkpoints.tgz
  ```
- (Optional) Now we can run the demo. To detect the sample image `$SQDT_ROOT/data/sample.png`,

  ```Shell
  cd $SQDT_ROOT/
  python ./src/demo.py
  ```
  If the installation is correct, the detector should generate this image: ![alt text](https://github.com/BichenWuUCB/squeezeDet/blob/master/README/out_sample.png)

  To detect other image(s), use the flag `--input_path=./data/*.png` to point to input image(s). Input image(s) will be scaled to the resolution of 1242x375 (KITTI image resolution), so it works best when original resolution is close to that.  


- Now we can run the program for our files. Make sure you download photos into the "input_path"
  ```Shell
  python ./src/demo.py  --input_path="./../one_half_x/object-detection-crowdai/*.jpg"  --out_dir="result/"
  ```

## Parameters:
- Boxes, Labels, probabilities

  ```Shell
        print (final_boxes)
        print (final_probs)
        print (final_class)
        print (mc.CLASS_NAMES)
  ```
  
- Output

  ```Shell
  [array([ 948.63018799,  187.26040649,   34.27148438,  122.31188965], dtype=float32)]
  [0.4635523]
  [1]
  ['car', 'pedestrian', 'cyclist']

  ```
  
- Threshold

  ```Shell
  /squeezeDet/src/config/kitti_squeezeDet_config.py
  

  ```
  - Size
  ```
  kitti_squeezeDet_config.py
  mc.IMAGE_WIDTH           = 1248
  mc.IMAGE_HEIGHT          = 384
  


  ```
  - Flow chart
  ```
  train.py -> SqueezeDet(mc) @ kitti_squeezeDet_config.py -> 
  imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc) @ kitti.py -> imdb.__init__@imdb.py ->
  im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))@ imdb.py-> imdb.read_batch() @ imdb.py 
  train.py
    elif FLAGS.net == 'squeezeDet':
    mc = kitti_squeezeDet_config()  ```
  imdb.py
  im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
  
  demo_v6.py -> kitti_squeezeDet_config() kitti_squeezeDet_config.py
  ```

## Training/Validation:
- Download KITTI object detection dataset: [images](http://www.cvlibs.net/download.php?file=data_object_image_2.zip) [(direct link)](http://kitti.is.tue.mpg.de/kitti/data_object_image_2.zip) and [labels](http://www.cvlibs.net/download.php?file=data_object_label_2.zip) [(direct link)](http://kitti.is.tue.mpg.de/kitti/data_object_label_2.zip). Put them under `$SQDT_ROOT/data/KITTI/`. Unzip them, then you will get two directories:  `$SQDT_ROOT/data/KITTI/training/` and `$SQDT_ROOT/data/KITTI/testing/`. 

- Now we need to split the training data into a training set and a vlidation set. 

  ```Shell
  cd $SQDT_ROOT/data/KITTI/
  mkdir ImageSets
  cd ./ImageSets
  ls ../training/image_2/ | grep ".png" | sed s/.png// > trainval.txt
  ```
  `trainval.txt` contains indices to all the images in the training data. In our experiments, we randomly split half of indices in `trainval.txt` into `train.txt` to form a training set and rest of them into `val.txt` to form a validation set. For your convenience, we provide a script to split the train-val set automatically. Simply run
  
    ```Shell
  cd $SQDT_ROOT/data/
  python random_split_train_val.py
  ```
  
  then you should get the `train.txt` and `val.txt` under `$SQDT_ROOT/data/KITTI/ImageSets`. 
  
  (Note: Change image_set_dir = './KITTI/ImageSets' if your data is not in KITTI)

  When above two steps are finished, the structure of `$SQDT_ROOT/data/KITTI/` should at least contain:
  

  ```Shell
  $SQDT_ROOT/data/KITTI/
                    |->training/
                    |     |-> image_2/00****.png
                    |     L-> label_2/00****.txt
                    |->testing/
                    |     L-> image_2/00****.png
                    L->ImageSets/
                          |-> trainval.txt
                          |-> train.txt
                          L-> val.txt
  ```

- Next, download the CNN model pretrained for ImageNet classification:
  ```Shell
  cd $SQDT_ROOT/data/
  # SqueezeNet
  wget https://www.dropbox.com/s/fzvtkc42hu3xw47/SqueezeNet.tgz
  tar -xzvf SqueezeNet.tgz
  # ResNet50 
  wget https://www.dropbox.com/s/p65lktictdq011t/ResNet.tgz
  tar -xzvf ResNet.tgz
  # VGG16
  wget https://www.dropbox.com/s/zxd72nj012lzrlf/VGG16.tgz
  tar -xzvf VGG16.tgz
  ```

- Now we can start training. Training script can be found in `$SQDT_ROOT/scripts/train.sh`, which contains commands to train 4 models: SqueezeDet, SqueezeDet+, VGG16+ConvDet, ResNet50+ConvDet. 
  ```Shell
  cd $SQDT_ROOT/
  ./scripts/train.sh -net (squeezeDet|squeezeDet+|vgg16|resnet50) -train_dir /tmp/bichen/logs/squeezedet -gpu 0
  ```

  Training logs are saved to the directory specified by `-train_dir`. GPU id is specified by `-gpu`. Network to train is specificed by `-net` 
- We choose the following

  ```Shell
  ./scripts/train.sh -net squeezeDet -train_dir /tmp/bichen/logs/squeezedet -gpu 0
  ```
### Validation version 1: 
Chenghung Yeh
- Check the checkpoint in the training path. Note: Use "model.ckpt-999999" instead of "model.ckpt-998000.data-00000-of-00001"

  ```Shell
  python ./src/demo_v5_crop_random_debug.py  --input_path="./../one_half_x/object-detection-crowdai/*.jpg"  --out_dir="result2/" --checkpoint="/tmp/bichen/logs/squeezedet/train/model.ckpt-999999"
  ```


### Validation version 2: 

- Before evaluation, you need to first compile the official evaluation script of KITTI dataset
  ```Shell
  cd $SQDT_ROOT/src/dataset/kitti-eval
  make
  ```

- Then, you can launch the evaluation script (in parallel with training) by 

  ```Shell
  cd $SQDT_ROOT/
  ./scripts/eval.sh -net (squeezeDet|squeezeDet+|vgg16|resnet50) -eval_dir /tmp/bichen/logs/squeezedet -image_set (train|val) -gpu 1
  ```

  Note that `-train_dir` in the training script should be the same as `-eval_dir` in the evaluation script to make it easy for tensorboard to load logs. 

  You can run two evaluation scripts to simultaneously evaluate the model on training and validation set. The training script keeps dumping checkpoint (model parameters) to the training directory once every 1000 steps (step size can be changed). Once a new checkpoint is saved, evaluation threads load the new checkpoint file and evaluate them on training and validation set. 

- Finally, to monitor training and evaluation process, you can use tensorboard by

  ```Shell
  tensorboard --logdir=$LOG_DIR
  tensorboard --logdir=/tmp/bichen/logs/squeezedet/train/
  ```
  Here, `$LOG_DIR` is the directory where your training and evaluation threads dump log events, which should be the same as `-train_dir` and `-eval_dir` specified in `train.sh` and `eval.sh`. From tensorboard, you should be able to see a lot of information including loss, average precision, error analysis, example detections, model visualization, etc.

  ![alt text](https://github.com/BichenWuUCB/squeezeDet/blob/master/README/detection_analysis.png)
  ![alt text](https://github.com/BichenWuUCB/squeezeDet/blob/master/README/graph.png)
  ![alt text](https://github.com/BichenWuUCB/squeezeDet/blob/master/README/det_img.png)
### Kitti Label [format](https://github.com/NVIDIA/DIGITS/blob/v4.0.0-rc.3/digits/extensions/data/objectDetection/README.md)

The format for KITTI labels is explained in the `readme.txt` from the "Object development kit".
Here is the relevant portion:
```
Data Format Description
=======================

The data for training and testing can be found in the corresponding folders.
The sub-folders are structured as follows:

  - image_02/ contains the left color camera images (png)
  - label_02/ contains the left color camera label files (plain text files)
  - calib/ contains the calibration for all four cameras (plain text file)

The label files contain the following information, which can be read and
written using the matlab tools (readLabels.m, writeLabels.m) provided within
this devkit. All values (numerical or strings) are separated via spaces,
each row corresponds to one object. The 15 columns represent:

#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.

Here, 'DontCare' labels denote regions in which objects have not been labeled,
for example because they have been too far away from the laser scanner. To
prevent such objects from being counted as false positives our evaluation
script will ignore objects detected in don't care regions of the test set.
You can use the don't care labels in the training set to avoid that your object
detector is harvesting hard negatives from those areas, in case you consider
non-object regions from the training images as negative examples.

The coordinates in the camera coordinate system can be projected in the image
by using the 3x4 projection matrix in the calib folder, where for the left
color camera for which the images are provided, P2 must be used. The
difference between rotation_y and alpha is, that rotation_y is directly
given in camera coordinates, while alpha also considers the vector from the
camera center to the object center, to compute the relative orientation of
the object with respect to the camera. For example, a car which is facing
along the X-axis of the camera coordinate system corresponds to rotation_y=0,
no matter where it is located in the X/Z plane (bird's eye view), while
alpha is zero only, when this object is located along the Z-axis of the
camera. When moving the car away from the Z-axis, the observation angle
will change.

To project a point from Velodyne coordinates into the left color image,
you can use this formula: x = P2 * R0_rect * Tr_velo_to_cam * y
For the right color image: x = P3 * R0_rect * Tr_velo_to_cam * y

Note: All matrices are stored row-major, i.e., the first values correspond
to the first row. R0_rect contains a 3x3 matrix which you need to extend to
a 4x4 matrix by adding a 1 as the bottom-right element and 0's elsewhere.
Tr_xxx is a 3x4 matrix (R|t), which you need to extend to a 4x4 matrix
in the same way!

Note, that while all this information is available for the training data,
only the data which is actually needed for the particular benchmark must
be provided to the evaluation server. However, all 15 values must be provided
at all times, with the unused ones set to their default values (=invalid) as
specified in writeLabels.m. Additionally a 16'th value must be provided
with a floating value of the score for a particular detection, where higher
indicates higher confidence in the detection. The range of your scores will
be automatically determined by our evaluation server, you don't have to
normalize it, but it should be roughly linear. If you use writeLabels.m for
writing your results, this function will take care of storing all required
data correctly.
```
