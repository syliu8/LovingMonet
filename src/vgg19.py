# ECBM 4040 Final Project CNN
"""
This .py file includes the architecture of the  VGG19 network: 

Date:
12/17/2017
"""

#import necessary library
import tensorflow as tf
import numpy as np
import scipy.io


class VGG:
    '''
    The class of VGG19 network
    '''
    # record the 19 layers in the pre-trained model

    def __init__(self, data_path):
        """
        Initilzed VGG class

        Parameters
        ----------
        data_path : str
            the path of the pretrained VGG network

        """
        self.data_path = data_path
        self.data = scipy.io.loadmat(data_path)
        # record the mean values for normalization
        mean = self.data['normalization'][0][0][0]
        self.mean_pixel = np.mean(mean, axis=(0, 1))
        self.weights = self.data['layers'][0]
    
    
    LAYERS = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
        )


    def normalize(self, image):
        """
        normalize the input image

        Parameters
        ----------
        image : np.array
           the image to be normalized
        Returns
        -------
        np.array
        the normalized np.array image

        """
        return image - self.mean_pixel

    def unnormalize(self, image):
        """
        unnormalize the normalized picture(recover)

        Parameters
        ----------
        image : np.array
           the image to be unnormalized
        Returns
        -------
        np.array
        the unnormalized image
        """
        return image + self.mean_pixel
        
    def vgg_net(self, input_image):
        """
        the vgg network

        Parameters
        ----------
        input_image : np.array
           the image to be calculated by the pre-trained model
        Returns
        -------
        dictionary 
        the key, value pair, with keys being the name of each layer, value being  the calculated results from each layer
        """
        # the returned dictionary
        curnet = {}
        # initialize the current layer by input imnage, then pass layer by layaer later
        current_layer = input_image
        
        # calculate the result of each layer using pre-tarined model
        for i, name in enumerate(self.LAYERS):
            # use conv2d for convolutional layers
            if (name[:4] == 'conv'):
                kernels, bias = self.weights[i][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                bias = bias.reshape(-1)
                conv = tf.nn.conv2d(current_layer, tf.constant(kernels), strides=(1, 1, 1, 1),padding='SAME')
                current_layer = tf.nn.bias_add(conv,bias)
                
            # use tf.nn.relu for relu layers
            elif (name[:4]=='relu'):
                current_layer = tf.nn.relu(current_layer)
            # use max_pool for pooling layers
            #    relu = tf.nn.max_pool(current_layer, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),padding='SAME')
 
            # use average_pool for pooling layers as suggested by Gatys
            elif (name[:4] == 'pool'):
                current_layer = tf.nn.avg_pool(current_layer, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),padding='SAME')
            
            #  record the result in the dictionary
            curnet[name] = current_layer

        return curnet

