## Content Table

0. Machine Learning - Giving Computers the Ability to **Learn from Data** 
0. Training Machine Learning Algorithms for **Classification** 
0. A Tour of Machine Learning Classifiers Using **Scikit-Learn** 
0. Building Good Training Sets – **Data Pre-Processing** 
0. Compressing Data via **Dimensionality Reduction** 
0. Learning Best Practices for **Model Evaluation and Hyperparameter Optimization** 
0. Combining Different Models for **Ensemble Learning** 
0. Applying Machine Learning to **Sentiment Analysis** 
0. Embedding a Machine Learning Model into a **Web Application**
0. Predicting Continuous Target Variables with **Regression Analysis** 
0. Working with Unlabeled Data – **Clustering Analysis** 
0. Training Artificial Neural Networks for **Image Recognition** 
0. Parallelizing Neural Network Training via **Theano** 

ch01

Building intelligent machines to transform data into knowledge
The three different types of machine learning
Making predictions about the future with supervised learning
Classification for predicting class labels
Regression for predicting continuous outcomes
Solving interactive problems with reinforcement learning
Discovering hidden structures with unsupervised learning
Finding subgroups with clustering
Dimensionality reduction for data compression
An introduction to the basic terminology and notations
A roadmap for building machine learning systems
Preprocessing - getting data into shape
Training and selecting a predictive model
Evaluating models and predicting unseen data instances
Using Python for machine learning
Installing Python packages
Summary

ch02

Artificial neurons - a brief glimpse into the early history of machine learning
Implementing a perceptron learning algorithm in Python
Training a perceptron model on the Iris dataset
Adaptive linear neurons and the convergence of learning
Minimizing cost functions with gradient descent
Implementing an Adaptive Linear Neuron in Python
Large scale machine learning and stochastic gradient descent
Summary


ch03

Choosing a classification algorithm
First steps with scikit-learn
Training a perceptron via scikit-learn
Modeling class probabilities via logistic regression
Logistic regression intuition and conditional probabilities
Learning the weights of the logistic cost function
Training a logistic regression model with scikit-learn
Tackling overfitting via regularization
Maximum margin classification with support vector machines
Maximum margin intuition
Dealing with the nonlinearly separable case using slack variables
Alternative implementations in scikit-learn
Solving nonlinear problems using a kernel SVM
Using the kernel trick to find separating hyperplanes in higher dimensional space
Decision tree learning
Maximizing information gain – getting the most bang for the buck
Building a decision tree
Combining weak to strong learners via random forests
K-nearest neighbors – a lazy learning algorithm
Summary

ch04

Dealing with missing data
Eliminating samples or features with missing values
Imputing missing values
Understanding the scikit-learn estimator API
Handling categorical data
Mapping ordinal features
Encoding class labels
Performing one-hot encoding on nominal features
Partitioning a dataset in training and test sets
Bringing features onto the same scale
Selecting meaningful features
Sparse solutions with L1 regularization
Sequential feature selection algorithms
Assessing feature importance with random forests
Summary

ch05

Unsupervised dimensionality reduction via principal component analysis 128
Total and explained variance
Feature transformation
Principal component analysis in scikit-learn
Supervised data compression via linear discriminant analysis
Computing the scatter matrices
Selecting linear discriminants for the new feature subspace
Projecting samples onto the new feature space
LDA via scikit-learn
Using kernel principal component analysis for nonlinear mappings
Kernel functions and the kernel trick
Implementing a kernel principal component analysis in Python
Example 1 – separating half-moon shapes
Example 2 – separating concentric circles
Projecting new data points
Kernel principal component analysis in scikit-learn
Summary


ch06

Streamlining workflows with pipelines
Loading the Breast Cancer Wisconsin dataset
Combining transformers and estimators in a pipeline
Using k-fold cross-validation to assess model performance
The holdout method
K-fold cross-validation
Debugging algorithms with learning and validation curves
Diagnosing bias and variance problems with learning curves
Addressing overfitting and underfitting with validation curves
Fine-tuning machine learning models via grid search
Tuning hyperparameters via grid search
Algorithm selection with nested cross-validation
Looking at different performance evaluation metrics
Reading a confusion matrix
Optimizing the precision and recall of a classification model
Plotting a receiver operating characteristic
The scoring metrics for multiclass classification
Summary

ch07

Learning with ensembles
Implementing a simple majority vote classifier
Combining different algorithms for classification with majority vote
Evaluating and tuning the ensemble classifier
Bagging – building an ensemble of classifiers from bootstrap samples
Leveraging weak learners via adaptive boosting
Summary


ch08

Obtaining the IMDb movie review dataset
Introducing the bag-of-words model
Transforming words into feature vectors
Assessing word relevancy via term frequency-inverse document frequency
Cleaning text data
Processing documents into tokens
Training a logistic regression model for document classification
Working with bigger data – online algorithms and out-of-core learning
Summary


ch09

Chapter 8 recap - Training a model for movie review classification
Serializing fitted scikit-learn estimators
Setting up a SQLite database for data storage Developing a web application with Flask
Our first Flask web application
Form validation and rendering
Turning the movie classifier into a web application
Deploying the web application to a public server
Updating the movie review classifier
Summary


ch10

Introducing a simple linear regression model
Exploring the Housing Dataset
Visualizing the important characteristics of a dataset
Implementing an ordinary least squares linear regression model
Solving regression for regression parameters with gradient descent
Estimating the coefficient of a regression model via scikit-learn
Fitting a robust regression model using RANSAC
Evaluating the performance of linear regression models
Using regularized methods for regression
Turning a linear regression model into a curve - polynomial regression
Modeling nonlinear relationships in the Housing Dataset
Dealing with nonlinear relationships using random forests
Decision tree regression
Random forest regression
Summary


ch11

Grouping objects by similarity using k-means
K-means++
Hard versus soft clustering
Using the elbow method to find the optimal number of clusters
Quantifying the quality of clustering via silhouette plots
Organizing clusters as a hierarchical tree
Performing hierarchical clustering on a distance matrix
Attaching dendrograms to a heat map
Applying agglomerative clustering via scikit-learn
Locating regions of high density via DBSCAN
Summary


ch12

Modeling complex functions with artificial neural networks
Single-layer neural network recap
Introducing the multi-layer neural network architecture
Activating a neural network via forward propagation
Classifying handwritten digits
Obtaining the MNIST dataset
Implementing a multi-layer perceptron
Training an artificial neural network
Computing the logistic cost function
Training neural networks via backpropagation
Developing your intuition for backpropagation
Debugging neural networks with gradient checking
Convergence in neural networks
Other neural network architectures
Convolutional Neural Networks
Recurrent Neural Networks
A few last words about neural network implementation
Summary


ch13

Building, compiling, and running expressions with Theano
What is Theano?
First steps with Theano
Configuring Theano
Working with array structures
Wrapping things up – a linear regression example
Choosing activation functions for feedforward neural networks
Logistic function recap
Estimating probabilities in multi-class classification via the softmax function
Broadening the output spectrum by using a hyperbolic tangent
Training neural networks efficiently using Keras
Summary






