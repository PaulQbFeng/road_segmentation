Those are the different steps to follow to train the CNN used our road segmentation project.

**Libraries**

 The following libraries have to be installed to be able to run the project:  
 
  - Keras      2.1.2  
  - CudNN      6.0  
  - Cuda       8.0  
  - Tensorflow 1.4.0  
  - h5py       2.7.1  
  - cython     0.27.3 
  
We create our neural net with Keras and hpy5 allows us to save the weights of the model and to load them back in the convolutional neural network.

**Instalation process**

To install all these libraries you first download and install Cuda 8.0 from https://developer.nvidia.com/cuda-downloads
Then download CudNN 6.0 from https://developer.nvidia.com/rdp/cudnn-download  
You now have to put the cudnn C:\...\cuda_v6\bin in your path environement variable.  
We assume that you have conda already installed.  
Now on the terminal run the following commands (we want to use gpu, in order to use cpu it's different command):  
C:> conda create -n tensorflow python=3.5  
C:> activate tensorflow  
(tensorflow)C:> pip install --ignore-installed --upgrade tensorflow-gpu'  
 
At this point you have Cuda, CudNN and Tensorflow working.  
Now install Keras with command    
C:> pip install keras     
C:> pip install h5py                                                                                                          <br>C:> pip install cython       
(this last command is if you use windows, for linux please refer to this forum https://github.com/keras team/keras/issues/3426)    

**Setup**

Computer: Acer Aspire VN7-592G  
OS: Windows 10  
CPU: Intel i7-6700HQ 2.6GHz  
RAM: 32 GB  
GPU: NVIDIA GeForce GTX 960M  

How to run:  
In order to run the code faster we already provide you the weights that made our result  
(the file 'weights-best-submission.hdf5') then directory hierarchy in order to run 'run.py' have to be  
Project  
 |  
 |--helpers.py  
 |--image_augmentation.py  
 |--mask_to_submission.py  
 |--prediction_with_weights.py  
 |--run.py  
 |--training.py  
 |--weights-best-submission.hdf5  
 |--window_patch_extend.py  
 |  
 +--test_set_images  
 | |  
 | +--test_1  
 | | |  
 | | |--test_1.png  
 | +--test_2  
 | | |
 | | |--test_2.png  
 ...  
 | +--test_50  
 | | |  
 | | |--test_50.png  
 +--training  
 | |  
 | +--groundtruth  
 | | |  
 | | |--satImage_001.png  
 | | |--satImage_002.png  
 ...  
 | | |--satImage_100.png  
 | +--images  
 | | |  
 | | |--satImage_001.png  
 | | |--satImage_002.png  
 ...  
 | | |--satImage_100.png  

When run.py in run, a new directory "predictions" should have been created, and a csv file "cnn_1_submission.csv" which corresponds to our best submission on Kaggle  

Description of the files:  

- helpers.py : this file contains the functions to load and save images    
- image_augmentation.py : this file is runnable and it creates a new folder "augmented_training", containing the augmented data (rotated and flipped pictures)   
- mask_to_submission.py : this file is runnable and creates a submission file ("cnn_1_submission.csv"), this will run only if the "prediction" folder has been created previously, and if it contains the predictions for test images  
- prediction_with_weights.py : this file is runnable and is expecting an argument which is the name of the weights file, it creates the predictions images in a new folder "predictions"  
- run.py : this file is runnable and only runs prediction_with_weights with our best weights, and then runs mask_to_submission
- training.py : this file is runnable and creates a model from the augmented data, the "augmented_training" folder has to exist. it takes some time to run (approximately 8 minutes per epoch on our setup, times 30 epochs)  
- weights-best-submission.hdf5 : this contains the weights that produce the best submission on Kaggle on our neural network  
- window_patch_extend.py : this file contains helper functions to extend an image, create windows and patches.  

Workflow for full training and predicting:  

python image_augmentation.py  
python training.py  
python prediction_with_weights.py weights.hdf5  
python mask_to_submission  
