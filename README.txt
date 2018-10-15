# Project Title: deep learning for arts

Using the methodology discussed in the one paper and a pre-trained VGG network, we are able to separate and recombine content and style of arbitrary images, to generate artistic images of high perceptual quality. Besides, we also perfrom segmented style transfer and video transfer.

### Prerequisites

Python 3 with the following libraries installed:
-tensorflow
-matplotlib
-numpy
-scipy
-importlib
-PIL
-cv2
-shutil
-functools

### Installing

install the necessary package and prepare necessary pictures to be transfered

## Running the tests
-copy the libraries/pictures into the working dictionary
-to transfer single picture/video, open "libs/main(pic+video).ipynb", run the ipny file
-to perform segmentation,firstly transfer the whole single picture, and then save the picture to pics, and then put the original picture and the transfered picture to input folder, run "libs/segmentation_art-transfer.ipynb"


