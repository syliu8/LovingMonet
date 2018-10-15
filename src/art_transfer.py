# ECBM 4040 Final Project CNN
"""
This .py file includes two classes for Columbia ECBM 4040 final project: 
1.transfer_art
2.calculate_loss

Date:
12/17/2017
"""

#Import necessary libraries
from vgg19 import VGG
import tensorflow as tf
import numpy as np
from sys import stdout
from functools import reduce

class transfer_art:
    """
    this class is the core part of extraing style and content, it includes the training process from the white noise pictures  
      """

    content_layer = 'conv4_2'
    # we can change this part when we want to see the effects of only using part of the layers
    art_layers = (('conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'))
    
    def __init__(self, vgg_path, content,
                art, content_weight,
                art_weight, tv_weight,
                initial,
                gpu_device):
        """
         initialize the transfer_art class

         Parameters
         ----------
         vgg_path : str
           the directory of the pretrained vgg path
           
         content: np.array
          the content image
         
         art: np.array
          the style image
         
         content_weight:float
         the alpha in the  model, the parameters to penalize the content loss term
         
         art_weight:
         the beta in the  model, the parameters to penalize the style loss term
         
         tv_weight:
         the gamma in the  model, the parameters to penalize the total varational loss term
         
         initial: np.array/none
         the initial image, if  none, the algorithm will start from a default white noise image
         
         gpu_device: float('/gpu:0' or '/cpu:0')
         control whether the device is gpu/cpu
       """
        with tf.device(gpu_device):
            # read the pretrained model
            self.vgg = VGG(vgg_path)
            # read two pictures
            self.content = content
            self.art = art
            self.image = self.get_white_image(initial)
            #  use the calculate_loss class to calculate total loss
            loss_calculator = calculate_loss(self.vgg, self.image);
            #  calculate the loss seprately
            self.content_loss = loss_calculator.content_loss(content,self.content_layer,content_weight)
            self.art_loss = loss_calculator.art_loss(art,self.art_layers,art_weight)
            self.total_variation_loss = loss_calculator.tv_loss(self.image,self.content.shape,tv_weight)
            # total loss
            self.loss = self.content_loss + self.art_loss + self.total_variation_loss
            
    def get_white_image(self,initial):
        """
        generate the white noise pictures

        Parameters
        ----------
        initial: np.array/none
         the initial image, if  none, the algorithm will start from a default white noise image
        Returns
        -------
        np.array
         tf.variable(np.array)

        """
        if initial is None:
            initial_image = tf.random_normal(self.content.shape)
        else:
            initial_image = self.vgg.normalize(initial)
        return tf.Variable(initial_image)
    
    def current_loss(self):
        """
        store the current loss

        Parameters
        ----------
        initial: np.array/none
         the initial image, if  none, the algorithm will start from a default white noise image
        Returns
        -------
        np.array
         tf.variable(np.array)


        """
        losses = {}
        losses['content'] = self.content_loss.eval()
        losses['art'] = self.art_loss.eval()
        losses['total_variation'] = self.total_variation_loss.eval()
        losses['total'] = self.loss.eval()
        return losses

    def train(self, learning_rate, iterations, checkpoint_iterations):
        """
        the traning process

        Parameters
        ----------
        learning_rate : float
           the learning rate of the algorithms
        iterations: int
            the total iteration times
        checkpoint_iterations: int
           the  frequency of reproting the total loss
       
        Returns
        -------
        (iteration, image, loss)

        """

        # for futher facilating the trainig process
        def is_checkpoint_iteration(i):
            return (checkpoint_iterations and i % checkpoint_iterations == 0) or i == iterations - 1
        
        # optimizer setup
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        #initialize the loss
        best_loss = float('inf')
        best = None
        
        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer())
            
            for i in range(iterations):
                stdout.write('Iteration %d/%d\n' % (i + 1, iterations))
                train_step.run()

                if is_checkpoint_iteration(i):
                    current_loss = self.loss.eval()
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best = self.image.eval()
                    yield (
                        (None if i == iterations - 1 else i),
                        self.vgg.unnormalize(best.reshape(self.content.shape[1:])),
                        self.current_loss()
                    ) 
                    
                    
# class transfer_art

                 
class calculate_loss:
    """
    the class to  calculate total loss
    """            
    def __init__(self, vgg, artified_image):
        """
         initialize the calculate loss class

         Parameters
         ----------
         vgg:  object,
             the pre-trained model
         artified_image: np.array
             the image of which loss is calculated
         
        """
        self.vgg = vgg
        self.net = vgg.vgg_net(artified_image)

    def content_loss(self, content, content_layer, content_weight):
        
        """
         calculate the content loss

         Parameters
         ----------
         content: np.array
          content image
         content_layer: str
          name of content layer
         Returns
         -------
         float
         the calculated content loss
        """        
        content_image = tf.placeholder('float', shape=content.shape)
        content_net = self.vgg.vgg_net(content_image)

        with tf.Session() as sess:
            content_feature = content_net[content_layer].eval(
                    feed_dict={content_image: self.vgg.normalize(content)})
            
        # content loss
        content_loss = content_weight * (2 * tf.nn.l2_loss(
                self.net[content_layer] - content_feature) /
                content_feature.size)

        return content_loss
    
    def art_loss(self, art, art_layers, art_weight):
        """
         calculate the style loss

         Parameters
         ----------
         art: np.array
          style image
         art_layers: str
          name of art layer
         Returns
         -------
         float
         the calculated content loss
        """         
        image = tf.placeholder('float', shape=art.shape)
        art_net = self.vgg.vgg_net(image)

        with tf.Session() as sess:
            art_normalized = self.vgg.normalize(art)
            #intialize
            art_loss = 0

            for layer in art_layers:
                art_image_gram = self._calculate_art_gram_matrix_for(art_net,image,layer,art_normalized)                                           # clculate the gram matrix                                                                     
                input_image_gram = self._calculate_gram_matrix(layer)
                art_loss += art_weight * (2 * tf.nn.l2_loss(input_image_gram - art_image_gram) / art_image_gram.size)

        return art_loss

    def tv_loss(self, image, shape, tv_weight):
        """
         calculate the total varational loss

         Parameters
         ----------
         image: np.array
          input image
         shape: np.array
          shape of input image
         tv_weight: float
          weight of total variation, gamma in the formula
         Returns
         -------
         float
         the calculated total varational loss
        """        
        # total variation denoising
        
        tv_y_size = _tensor_size(image[:,1:,:,:])
        tv_x_size = _tensor_size(image[:,:,1:,:])
        tv_loss = tv_weight * 2 * (
                (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /
                    tv_y_size) +
                (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /
                    tv_x_size))

        return tv_loss

    def _calculate_art_gram_matrix_for(self, network, image, layer, art_image):
        """
         initialize the calculate loss class

         Parameters
         ----------
         image : np.array
             the image to be normalized
         Returns
         -------
          np.array
          the no
        """
        image_feature = network[layer].eval(feed_dict={image: art_image})
        image_feature = np.reshape(image_feature, (-1, image_feature.shape[3]))
        return np.matmul(image_feature.T, image_feature) / image_feature.size

    def _calculate_gram_matrix(self, layer):
        """
         initialize the calculate loss class

         Parameters
         ----------
         image : np.array
             the image to be normalized
         Returns
         -------
          np.array
          the no
        """        
        image_feature = self.net[layer]
        _, height, width, number = map(lambda i: i.value, image_feature.get_shape())
        size = height * width * number
        image_feature = tf.reshape(image_feature, (-1, number))
        return tf.matmul(tf.transpose(image_feature), image_feature) / size
def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)