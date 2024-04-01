# ML4Sci_DeepLense
### Common Test I. Multi-Class Classification
Task: Build a model for classifying the images into lenses using PyTorch or Keras. Pick the most appropriate approach and discuss your strategy.

Strategy:
- Our methodology begins with the utilization of the pretrained ResNet50 architecture, from which the classification head is removed, followed by the flattening of the architecture.

- Subsequently, the integration of batch normalization and dropout techniques is undertaken to enhance the model's generalization capabilities, albeit with a potential trade-off in early convergence.

- Lastly, a dense layer with 3 outputs is introduced, coupled with the application of softmax activation, to facilitate the generation of probabilities corresponding to the 3 classes.
 #### results:
 #### traning auc:0.9757
 #### validation auc:0.9768
<img src="Common Test 1. Multi-Class Classification/results/Screenshot 2024-03-27 200636.png">
<img src="Common Test 1. Multi-Class Classification/results/Screenshot 2024-03-27 200648.png">

### Specific Test II. Lens Finding
Task: Build a model classifying the images from \easy directory into lenses using PyTorch or Keras. Evaluate your algorithm on the images from \hard directory; note that only 20% of them have labels available, you can use the rest to train the domain adaptation model or other model of your choice. Pick the most appropriate approach and discuss your strategy.

Strategy:

- We need to combine the images from the three different channels into one image for image classification.

- The image branch consists of two Conv2D layers followed by a MaxPooling2D layer to extract features from the image data. Subsequently, the input features from the CSV file are processed through a separate branch with a dense layer, and the resulting features are concatenated with those from the image branch.

- Following the concatenation, a custom self-attention layer is applied to the combined output. This attention mechanism calculates attention weights for both the image and non-image features, enabling the model to emphasize the most relevant parts of the data. The weighted sum of these features is then computed.

-Finally, the output of the attention layer undergoes two Dense layers. The final layer employs a sigmoid activation function to produce a probability output.
 #### results:
 #### traning auc:1.0000
 #### validation auc:1.0000
<img src="Specific Test 2. Lens Finding/results/Screenshot 2024-03-28 010121.png">
<img src="Specific Test 2. Lens Finding/results/Screenshot 2024-03-28 010143.png">

### Specific Test III. Image Super-resolution
#### Task III.A:
Task III.A: Train a deep learning-based super resolution algorithm of your choice to upscale low-resolution strong lensing images using the provided high-resolution samples as ground truths. Please implement your approach in PyTorch or Keras and discuss your strategy.

Strategy:
### SuperResCNN (Super-Resolution Convolutional Neural Network)
- Before attempting complex models on this task, we must try to establish a baseline model and analyze 


- The most optimal way to start is by implementing the SuperResCNN model that implements an upsampling layer followed by a three-layered neural network to learn the mapping between low-resolution and high-resolution images, such that the first layer can capture the low-level features, the second layer can capture high level features and the final layer can reconstruct the high resolution image

<img src="Specific Test 3. Image Super-resolution/Task III.A/results/Screenshot 2024-04-01 153727.png">

### EDSR (Enhanced Deep Residual Networks)
- Since the above SuperResCNN model has performed well, there could be a chance of improvement by training an EDSR model first (which has residual blocks that can help capture more complex image features) and then only if it works well, we can perform additional steps like cascading to take the output of the SuperResCNN and feed it to the EDSR.

<img src="Specific Test 3. Image Super-resolution/Task III.A/results/Screenshot 2024-04-01 153738.png">

### ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks)
- Generative Adversarial Networks can result in extremely good results if problems such as mode collapse can be handled using appropriate loss functions.

- We can use Residual Dense Blocks (RRDB) to extract high-level features from the input image. A perceptual loss function can be used that measures the similarity between the high-resolution image generated by the generator and the ground truth high-resolution image. It is based on the VGG19 network and computes the mean squared error between the feature maps of the generated and ground truth images at a selected layer. This loss function encourages the generator to produce images that are visually similar to the ground truth images.

- we can use sub-pixel convolution for image upscaling, which enables the network to generate high-resolution images with finer details. This technique involves reshaping the features extracted from the previous layer into a tensor with a higher spatial resolution and then applying convolutional layers to produce the final high-resolution image.

<img src="Specific Test 3. Image Super-resolution/Task III.A/results/Screenshot 2024-04-01 153800.png">

 #### results:

 | Model      | MSE        | SSIM       | PSNR       |
|------------|------------|------------|------------|
| SuperResCNN (Super-Resolution Convolutional Neural Network) | 5.9711   | 0.999998    | 41.97 |
| EDSR (Enhanced Deep Residual Networks)       | 0.00094   |0.999   | 31.995  |
| ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks)     | 0.00053   | 0.99994   | 30.7932  |

#### Task III.B:

Strategy:

#### SuperResCNN (Super-Resolution Convolutional Neural Network)
- In Task 3a, compared to all the resolution models above, SuperResCNN performed better. Therefore, we will use the same model for the task.

<img src="Specific Test 3. Image Super-resolution/Task III.B/results/Screenshot 2024-04-01 203209.png">

#### FSRCNN(Fast Super-Resolution Convolutional Neural Network)
- The SuperResCNN model is not effective for the given task. Instead, we will use the FERCNN mode
- FSRCNN has a relatively shallow network, which allows us to understand the effect of each component more easily. It is even faster and produces better reconstructed image quality compared to the previous SRCNN model.

  <img src="Specific Test 3. Image Super-resolution/Task III.B/results/Screenshot 2024-04-01 203221.png">

  #### results:

 | Model      | MSE        | SSIM       | PSNR       |
|------------|------------|------------|------------|
| SuperResCNN (Super-Resolution Convolutional Neural Network) | 0.001934   | 0.999   | 18.907 |
| FSRCNN(Fast Super-Resolution Convolutional Neural Network)      |  0.0017   | 0.999   | 23.6147 |


### Specific Test VI. SSL on Real Dataset

Task: Build a Self-Supervised Learning model for classifying the images into lenses using PyTorch or Keras. Pick the most appropriate approach and discuss your strategy

Strategy: 

- After implementing an Equivariant Transformer architecture featuring custom RotationalConv2D layers designed for maintaining equivariance with respect to input rotations, we augment the model with a convolution operation applied to rotated inputs, followed by combining resulting feature maps through element-wise addition. Additionally, we leverage pre-trained ResNet50 weights for expedited learning of representations.

- For the loss function, we opt for contrastive loss, which incentivizes the model to generate embeddings that are closer for similar images and farther apart for dissimilar ones. Specifically, we compute the sum of squared distances between positive pairs (where y_true = 1) and the squared hinge loss between negative pairs (where y_true = 0) with a designated margin.

Subsequently, post-training the embedding model, we fine-tune it for our classification task.
<img src="Specific Test 4. SSL on Real Dataset/results/Screenshot 2024-03-27 120321.png">

