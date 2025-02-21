# Angle Loss project

## General description

Hi all, this project aims to implement some Angle Loss concepts from different papers.
The core idea of Angle Loss is: 
"
We have a task, let's say classification, and a dataset, let's say animals images.
The issue with the data is that the images are very, very similar between one another, though they belong to different
classes. So, we want to create a features extractor model, trained with an **Angle Loss**, whose objective is to create 
**discriminative** features such that data samples from different classes are pushed apart from each other and samples of the same class
gets close to each other.
"
And that's it for the Angle loss main idea. A note: in the example mentioned above (classification), the model will be trained with the
Angle loss to obtain a feature extractor. Then, the feature extractor gets frozen and attached to a new model ending with a classic
softmax layer to classify the data during training.

This project is implemented in Python with Tensorflow.

## Repo folders description

- This repo contains an example of classification a task with classic a benchmark dataset (MNIST), "/examples".
- The core **Angle Loss** ideas are implemented in "/src/models/angle_margin". So, if you want to look at an example
of how to implement Angle losses here is where you need to look.
- This project uses Tensorboard to plot some useful insight about the training with Angle loss implemented
in the "/examples" section.

# References