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

### Specific Test VI. SSL on Real Dataset

Task: Build a Self-Supervised Learning model for classifying the images into lenses using PyTorch or Keras. Pick the most appropriate approach and discuss your strategy

Strategy: 

- After implementing an Equivariant Transformer architecture featuring custom RotationalConv2D layers designed for maintaining equivariance with respect to input rotations, we augment the model with a convolution operation applied to rotated inputs, followed by combining resulting feature maps through element-wise addition. Additionally, we leverage pre-trained ResNet50 weights for expedited learning of representations.

- For the loss function, we opt for contrastive loss, which incentivizes the model to generate embeddings that are closer for similar images and farther apart for dissimilar ones. Specifically, we compute the sum of squared distances between positive pairs (where y_true = 1) and the squared hinge loss between negative pairs (where y_true = 0) with a designated margin.

Subsequently, post-training the embedding model, we fine-tune it for our classification task.
<img src="Specific Test 4. SSL on Real Dataset/results/Screenshot 2024-03-27 120321.png">

