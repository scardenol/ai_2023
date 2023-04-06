# ai_2023
Repo for the workshops of the AI Course for 2023-1

# Projects
The repo is structured in such a way that every folder of the main directory contains an specific workshop or main project. Each workshop required specific tasks with specific algorithms that are stored as subfolders.

# 1. Supervised
This directory contains the first workshop, where a custom multi-layer perceptron (MLP) was developed from scratch. Temporal data with 2 inputs and 1 output was considered to investigate the relationship between neural network architecture and the progress of training and learning in a regression task.

# 2. Hybrid supervised
This directory contains the second workshop, where we focused on the implementation of autoencoders, convolutional neural networks (CNN) and generative adversarial networks (GAN).

# 3. Unsupervised
This directory contains the third workshop, where we performed exploratory clustering with density-based methods (mountain and substractive) to determine the optimal number of clusters $k$ for an arbitrary data set. Afterwards, the resulting optimal number of clusters k was passed as an hyperparameter for distance-based methods ($k$-means and fuzzy $c$-means) to compute the membership or clusterize the data. This workflow was implemented with the original data space, a high-dimensional space with the use of an autoencoder, and a low-dimensional space obtained by embedding tha original data with UMAP. We calculated internal and external validation indexes, and we also considered multiple hyperparameter configurations for both the density-based and distance-based methods.
