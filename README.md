# Vision-Transformer-Reproduction
## Abstract
This work is maninly about the reproduction of the Vision Transformer Model.  
The purpose of the project is to build a Vit model based on the paper **AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE**, and apply the model to solve the classify problem for the MNIST dataset.

## Code 
The code was divided into four part, and each of them take response for a specific function to increase the readability and maintainability of the code.  
### model.py
![image](https://github.com/A11enW/Vision-Transformer-Reproduction/assets/163519427/2a5a987a-7417-4fce-9f75-5b44efaf5706)
Realize the model in the figure.  
1. Pre=process: Cut the input figure into patches, adding the position_embedding and cls_token which are necessary in Transformer model.  
2. Transformer Encoder: Send the pre-processed data (Embedded Patches) into the transformer Encoder.  
3. Classification: Send the output from the Transformer into a MLP to realize the classification.  
### dataset.py
Obataining the training set, test set (Validation) and Submission date.  
*Since the submission set is obtained from Cargo, the accuracy will be given automatically when submite the result. In this case, the MNISTSubmissionDataset dont need label.
### utils.py
Load the dataset.  
### train.py
Call the other three parts of code and realize the training of the Vit and give the feedback about the model accuracy on the training and Submission dataset in each epoch.  
## Result Analysis
