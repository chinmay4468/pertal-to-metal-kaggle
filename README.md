# pertal-to-metal-kaggle
The Challenge
There are over 5,000 species of mammals, 10,000 species of birds, 30,000 species of fish – and astonishingly, over 400,000 different types of flowers.

In this competition, you’re challenged to build a machine learning model that identifies the type of flowers in a dataset of images (for simplicity, we’re sticking to just over 100 types).

Personal Contribution:
 # Experimenting with CNN Model: Impact of Image Size on Accuracy and Processing Time

In this analysis, we explore the performance of Convolutional Neural Network (CNN) models for image classification tasks. Two different image sizes, 224x224 pixels and 512x512 pixels, were considered. The goal was to evaluate the impact of varying model architectures and hyperparameters on classification accuracy.

## Experimental Setup

- **Image Sizes:**
  - 224x224 pixels
  - 512x512 pixels

- **Model Architecture:** We used a CNN model with multiple convolutional layers, max-pooling, batch normalization, and dropout for regularization.

- **Training Data:** The dataset consists of flower images categorized into different classes.

## Results

### Accuracy Comparison

After training the CNN model on both 192x192 and 512x512 pixel images, we observed the following accuracy results:

- **224x224 Pixels:**
  - Training Accuracy: [0.360414057970047, 0.35986512899398804, 0.3621392846107483, 0.36402133107185364, 0.360492467880249] <br>
  - Training Loss: [2.406597137451172, 2.405608654022217, 2.4017152786254883, 2.397501230239868, 2.40336537361145] <br>
  - Validation Accuracy: [0.2920258641242981, 0.2920258641242981, 0.2920258641242981, 0.2920258641242981, 0.2920258641242981] <br>
  - Validation Loss: [2.733835220336914, 2.733835220336914, 2.733835220336914, 2.733835220336914, 2.733835220336914] <br>

- **224x224 Pixels with batch normalization:**
  - Training Accuracy: [0.061715807765722275, 0.06046110391616821, 0.06257841736078262, 0.062029484659433365, 0.06163739040493965]<br>
  - Training Loss: [4.174342155456543, 4.1768388748168945, 4.17475700378418, 4.174972057342529, 4.1769609451293945]<br>
  - Validation Accuracy: [0.06142241507768631, 0.06142241507768631, 0.06142241507768631, 0.06142241507768631, 0.06142241507768631]<br>
  - Validation Loss: [4.174548625946045, 4.174548625946045, 4.174548625946045, 4.174548625946045, 4.174548625946045]<br>

After Using Batch normalization , we could see that the training accuracy decreased , so we need to find better solution to improve the model

So we tried experimenting with the higher pixel resolution 512x512 which slightly improved the accuracy . The validation accuracy increases from 14.49% to 23.49% after 5 epochs

The next step was to improve the model :
So we used different activation function including 'relu', 'elu', and 'leaky_relu', are experimented with in convolutional layers.<br>
The model exhibits promising performance with early stopping after 13 epochs, achieving a validation accuracy of 45.31%.<br>
Further experimentation with activation functions may lead to improved accuracy.


### Observations

#### Accuracy Impact:

- The model trained on 512x512 pixel images achieved higher accuracy across all metrics compared to the model trained on 224X224 pixel images.

- The increased spatial resolution in 512x512 images allowed the model to capture finer details and nuances in the flower images, resulting in improved classification performance.

#### Processing Time:

- While the 512x512 model showed superior accuracy, it came at the cost of increased processing time during training and inference.

- Larger image sizes result in more computations, leading to longer training times and slower inference. This is a trade-off between accuracy and computational efficiency.

## Conclusion

The choice of image size significantly influences the performance of a CNN model. Larger image sizes can enhance accuracy, but it's crucial to consider the computational resources and time constraints.

In practical scenarios, it's essential to strike a balance between achieving high accuracy and optimizing processing time based on the specific requirements of the application.
## References:
https://www.tensorflow.org/tutorials/load_data/tfrecord <br>
https://www.tensorflow.org/tutorials/ <br>
https://medium.com/machine-learning-researcher/convlutional-neural-network-cnn-2fc4faa7bb63 <br>
https://www.kaggle.com/code/pavansanagapati/a-simple-cnn-model-beginner-guide <br>
https://www.datacamp.com/tutorial/cnn-tensorflow-python


some Section of code is referenced from : https://www.kaggle.com/code/ryanholbrook/create-your-first-submission <br>
Also referred Submission from https://www.kaggle.com/competitions/tpu-getting-started <br>


