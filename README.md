# ML4Sci_DeepLense
### Common Test I. Multi-Class Classification
Task: Build a model for classifying the images into lenses using PyTorch or Keras. Pick the most appropriate approach and discuss your
strategy.
- Our methodology begins with the utilization of the pretrained ResNet50 architecture, from which the classification head is removed, followed by the flattening of the architecture.

- Subsequently, the integration of batch normalization and dropout techniques is undertaken to enhance the model's generalization capabilities, albeit with a potential trade-off in early convergence.

- Lastly, a dense layer with 3 outputs is introduced, coupled with the application of softmax activation, to facilitate the generation of probabilities corresponding to the 3 classes.
 #### results:
 #### traning auc:
 #### validation auc:
<img src="Common Test 1. Multi-Class Classification/results/Screenshot 2024-03-27 200636.png">
<img src="https://github.com/saibharath29/ML4Sci_DeepLense/blob/main/Common%20Test%201.%20Multi-Class%20Classification/results/Screenshot%202024-03-27%20111132.png">
