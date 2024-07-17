Image Classification Project

This project aims to classify images into one of six categories: Mountain, Icebergs, Forest, Sea, Building, and Street. A Convolutional Neural Network (CNN) is utilized to achieve high accuracy in identifying the appropriate category for each image. The project includes training the model, saving the best checkpoints, and deploying a web application using Flask where users can upload an image and receive the predicted category.

Project Overview
Data Preparation:
The dataset consists of images categorized into six folders: Mountain, Icebergs, Forest, Sea, Building, and Street.
Model Architecture:
A CNN model is constructed and trained on the labeled dataset to learn the features of each category.
Model Training:

The model is trained, and during training, the best performing model checkpoints are saved.
Web Application:
A Flask application is created to provide an interface for users to upload images and get predictions. The application uses the trained model to classify the uploaded images.

This project uses PyTorch for building and training the CNN model.
The Flask framework is used for creating the web application.
