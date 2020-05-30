
We created a skin cancer image classification model for a school project. We used various techniques to eventually achieve 75.5% f1 score on test data. Dataset we used is known as mnist:HAM10000 but we worked on a subset of it where we had only 5 classes of cancers and very imbalanced dataset on our hand.


PREPROCESSING
-Augmentation
One of the ways that we tried is manual augmentation. In order to do that, first in our local computer we have resized all images into 256x256 pixels and saved it into another directory. We did this because we have realized that supplied images have different resolutions. After that, as first thing we get training images from the new directory and split them into train and validation sets. We took 20% of the training data as validation and 80% of it remained for training the models. Then, we applied transformation to training portion of the data.In order to perform augmentation we implemented a function which takes an image as its parameter and returns 10 different images which are transformed versions of the image in that parameter. As transformation we used rotation 90 degrees to left and right, rotation 60 degrees to right, shifting to right, left, up and down by 10 pixels, flipping the image vertically, average and gaussian blurring of the image. We used OpenCV for performing transformations which are supported by Python. Following images show the instances of classes before and after the augmentation process.

Finally we used the pickle package of Python in order to save training validation and testing data into a pickle file and uploaded it into Google Drive. So that, in Collab we read data from these files before we train the models.

-ImageDataGenerator 
Another instrument we have used to resolve the imbalanced training data problem was using Keras' ImageDataGenerator. In this method, we have applied a series of random transformations to our original data and replaced the new transformed batch with the original one. To have a better understanding of the transformation process you can check the images below.

	  	   	     
                Original Image 1            Transformed Image 1-1        Transformed Image 1-2         Transformed Image 1-3

	  	    	     
     Original Image 2             Transformed Image 2-1       Transformed Image 2-2      Transformed Image 2-3

Among other data augmentation methods, ImageDataGenerator provided the best scores. Also, it has a quicker transformation process than others.
                   
-SMOTE 
We also tried the SMOTE method for preprocessing the training portion of data. We tried to use imblearn’s over_sampling package for SMOTE operation. However, since there are a lot of images with size 256x256 in training data, Google Collab’s RAM restriction did not allow us to perform this operation and the session crashed. If we could use SMOTE, we believe it would increase performance of our models.
-Class weight

	We have also specified class weights for each class. Thus, minority classes(mostly class 4 in our case) had more effect which led our model to equally learn from all classes. To determine weights, we have used the sklearn utility function.

MODEL
-Manuel models
At first, we started with a manual model where we selected layers by ourselves and tried it without previous knowledge through recitation. With some tuning we managed to get a 61% f1 score on our test data.When we searched more about models, we thought that using transfer models will bring better generalization over data. 

For Image Recognition, we can use pre-trained models available in the Keras core library. We used pre-trained versions of the models, trained by ImageNet database. The models like VGG16, VGG19, Resnet50, InceptionResNetV2, Xception models. We obtain the top results by using DenseNet201, Xception, and InceptionResNetV2. DenseNet201 architecture densely connects all the layers. This means each layer receives inputs from all the proceeding layers. We have the best F1-score by using DenseNet201. Since it has advantages like alleviate the vanishing gradient problem, strengthen feature propagation, encourage feature reuse.   

After trying out some different models, most of them stuck at around %65 f1 score which was essential for us. When we analyzed scores of individual classes, we have seen that especially classes 4 and 5 have low f1 values in comparison to others. And then combine them through an ensemble method to fuse their opinions. We tried both hard voting and soft voting methods. In hard voting we took the mode of decisions each model gave us. By this way we increased our certainty but the same problems were still occurring. 

	
	Before soft voting, in order to improve the performance of our individual models, we generated new data that has two classes which are class 4 and others. We took the same amount of data from class 4, and the same amount from other classes in total. Then we labeled class 4 as 0 and others as 1 and splitted this data into its own train and validation. Finally, we tried to train our model based on this data, it had a 0.81 f1 score on its validation. However, when we tried this on the real validation data with 5 classes, the model had some problems. It predicted a lot of 0, which means class 4, however there were not that many occurrences of that class in real validation data. We found out this was caused by dead relu problems. After trying sigmoid and leaky relu, it was not sufficient enough to get high enough values in real application.




At soft voting, each model gave their probabilities over each possible outcome. After that instead of taking a mode we average the probabilities of predictions and select the highest probability. By this way a more confident answer on a particular outcome cannot be eliminated by not confident outcomes of other models. Hence, in our test we observed better results on the soft voting model. When we ensemble the predictions of the models which are transferred  from DenseNet201, Xception and InceptionResNetV2 networks, we have an increase in our performance. In the following image you can see the confusion matrix and classification report of the ensembled predictions on validation data. In addition, on the validation data the ensembled predictions increased the f1 score by 7%, and on test data when we submitted the predictions on Kaggle, our score increased to 75% from 72%.





In conclusion, the problem of predicting class 4 is being solved with these ensemble predictions. As we can see from the confusion matrix when a model predicts 4 it is more likely to belong to class 4. In addition to ensembling predictions of different models, each model is trained with a different proportion of class 4 in training data and different augmentations of class 4’s images. Some of them were exposed to more aggressive transformations while some of them used mild ones. This difference in distribution might have affected the number of predictions made by models and those which are not correct is eliminated in the ensemble process by having more confident predictions.
