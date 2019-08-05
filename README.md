# Encouraging dog adoption through convolutional neural networks

If you know me, you know that I am obsessed with dogs. Whenever I’m walking down a street, driving in my car, or just existing, I can spot dogs from a mile away. You could call it a sixth sense or an obsession, but either way, I. LOVE. DOGS. 

For my final capstone project at Metis, I wanted to take my love of dogs and do something productive with that. Knowing that approximately 3.3 million dogs enter animal shelters nationwide, a little less than half of those dogs actually are adopted and find a home, and another 20% of these dogs are euthanized, I wanted to create an application that would encourage pet adoption by creating a fun way to discover dogs looking for a home. That’s where Doggogänger comes in. 

Doggogänger is a Flask app that will predict what dog breed a human or dog most resembles and if you are interested, it will redirect you to PetFinder.com where you can find a dog that you or whoever you input most resembles. 


## Data

To create the app, I used a dog image dataset that contained 8,531 photos of a total of 133 breeds of dogs. In addition, because I wanted my app to be able to detect human faces and find the breed the human most resembles, I also used a human face database with 13,233 photos.


## Models  

With a clear intention in mind, I created three separate models. Because I needed my application to distinguish between dogs and humans. I created a face detector and a dog detector. Then, once identified, I would need to identify which breed is most similar to the input photo. 

### Face Detector
To do this, first, I created a face detector using OpenCV’s implementation of Haar feature-based cascade classifiers to detect human faces in photos. Haar feature-based cascade classifiers is an effective object detection method where a cascade function is trained from a lot of positive (images of faces) and negative images (images without faces). Because it is pre-trained, it is very easy to use and has an accuracy of 95%. When used on my human images, it had 100% accuracy.


![](https://paper-attachments.dropbox.com/s_5F95FD4F1224A6D8BB3647F3191FEC90ADBB796FA4B4F3B95A08EF639619FCA5_1564685693832_image.png)


### Dog Detector
Next, I used a pre-trained ResNet-50 network to detect dogs in the uploaded images. ResNet, which stands for Residual Networks, is a popular type of convolutional network that allows training of deep networks by constructing the network through modules called residual models. ResNet-50 is a 50 layer Residual Network and because it is pre-trained, the model has a predicted object class defined already. As such, it made detecting the dogs in the photos easier. When I tested the model on my dataset of both dogs and humans, the model had 100% accuracy for both. 

### Convolutional Neural Network
To actually identify which dog breed the human or dog most resembles, I used transfer learning on a ResNet-50 network base with customized classification layers. I loaded the ResNet-50 model and combined it with a global average pooling (GAP) layer, a densenet with ReLu activation function, a dropout layer, and a final densenet layer with softmax activation function. I decided to use a global average pooling layer to minimize overfitting by reducing the total number of parameters in the model. GAP layers are used to reduce the spatial dimensions of a three-dimensional tensor by performing a more extreme type of dimensionality reduction. For a ResNet-50 model, to get rid of dense layers altogether, the GAP layer is followed by one densely connected layer with a softmax activation function that yields the predicted object class which is what I did for my model. 


## How It Works

![](https://paper-attachments.dropbox.com/s_5F95FD4F1224A6D8BB3647F3191FEC90ADBB796FA4B4F3B95A08EF639619FCA5_1564694125691_image.png)


The application will take in a photo, checks against the face and dog detectors mentioned above. Then, it will load the pre-trained ResNet-50 base, pass the image through the base and the classification layers, and output the breed the human or dog most looks like. In the next section, you can see the application in action.


## Preview
Because the Heroku dyno does not support GPUs or offer GPU instances, I could not deploy my application for public use. I screenrecorded the app in action in the meantime while I continue to find a solution. (If anyone has any suggestions, please reach out :) )  

[![](https://paper-attachments.dropbox.com/s_5F95FD4F1224A6D8BB3647F3191FEC90ADBB796FA4B4F3B95A08EF639619FCA5_1565020738818_image.png)](https://youtu.be/xpNOhZgXZmM)


## Summary

In summary, I believe that this application is a great way to discover new breeds of dogs. Some you may have never even heard of before! I also believe that it is a great way to encourage pet adoption to see what dogs are available and up for adoption and help them find a home. 
