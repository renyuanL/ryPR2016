Preface

I probably don't need to tell you that machine learning has become one of the most
exciting technologies of our time and age. Big companies, such as Google, Facebook,
Apple, Amazon, IBM, and many more, heavily invest in machine learning research
and applications for good reasons. Although it may seem that machine learning has
become the buzzword of our time and age, it is certainly not a hype. This exciting
field opens the way to new possibilities and has become indispensable to our daily
lives. Talking to the voice assistant on our smart phones, recommending the right
product for our customers, stopping credit card fraud, filtering out spam from our
e-mail inboxes, detecting and diagnosing medical diseases, the list goes on and on.
If you want to become a machine learning practitioner, a better problem solver, or
maybe even consider a career in machine learning research, then this book is for you!
However, for a novice, the theoretical concepts behind machine learning can be quite
overwhelming. Yet, many practical books that have been published in recent years
will help you get started in machine learning by implementing powerful learning
algorithms. In my opinion, the use of practical code examples serve an important
purpose. They illustrate the concepts by putting the learned material directly into
action. However, remember that with great power comes great responsibility! The
concepts behind machine learning are too beautiful and important to be hidden in
a black box. Thus, my personal mission is to provide you with a different book; a
book that discusses the necessary details regarding machine learning concepts, offers
intuitive yet informative explanations on how machine learning algorithms work,
how to use them, and most importantly, how to avoid the most common pitfalls.
If you type "machine learning" as a search term in Google Scholar, it returns an
overwhelmingly large number-1,800,000 publications. Of course, we cannot discuss
all the nitty-gritty details about all the different algorithms and applications that have
emerged in the last 60 years. However, in this book, we will embark on an exciting
journey that covers all the essential topics and concepts to give you a head start in this
field. If you find that your thirst for knowledge is not satisfied, there are many useful
resources that can be used to follow up on the essential breakthroughs in this field.

If you have already studied machine learning theory in detail, this book will show
you how to put your knowledge into practice. If you have used machine learning
techniques before and want to gain more insight into how machine learning really
works, this book is for you! Don't worry if you are completely new to the machine
learning field; you have even more reason to be excited. I promise you that machine
learning will change the way you think about the problems you want to solve and
will show you how to tackle them by unlocking the power of data.
Before we dive deeper into the machine learning field, let me answer your most
important question, "why Python?" The answer is simple: it is powerful yet very
accessible. Python has become the most popular programming language for data
science because it allows us to forget about the tedious parts of programming and
offers us an environment where we can quickly jot down our ideas and put concepts
directly into action.

Reflecting on my personal journey, I can truly say that the study of machine learning
made me a better scientist, thinker, and problem solver. In this book, I want to
share this knowledge with you. Knowledge is gained by learning, the key is our
enthusiasm, and the true mastery of skills can only be achieved by practice. The road
ahead may be bumpy on occasions, and some topics may be more challenging than
others, but I hope that you will embrace this opportunity and focus on the reward.
Remember that we are on this journey together, and throughout this book, we will
add many powerful techniques to your arsenal that will help us solve even the
toughest problems the data-driven way.

What this book covers

Chapter 1, Giving Computers the Ability to Learn from Data, introduces you to the
main subareas of machine learning to tackle various problem tasks. In addition, it
discusses the essential steps for creating a typical machine learning model building
pipeline that will guide us through the following chapters.

Chapter 2, Training Machine Learning Algorithms for Classification, goes back to
the origin of machine learning and introduces binary perceptron classifiers and
adaptive linear neurons. This chapter is a gentle introduction to the fundamentals
of pattern classification and focuses on the interplay of optimization algorithms and
machine learning.

Chapter 3, A Tour of Machine Learning Classifiers Using Scikit-learn, describes the
essential machine learning algorithms for classification and provides practical
examples using one of the most popular and comprehensive open source machine
learning libraries, scikit-learn.

Chapter 4, Building Good Training Sets – Data Preprocessing, discusses how to deal with
the most common problems in unprocessed datasets, such as missing data. It also
discusses several approaches to identify the most informative features in datasets
and teaches you how to prepare variables of different types as proper inputs for
machine learning algorithms.

Chapter 5, Compressing Data via Dimensionality Reduction, describes the essential
techniques to reduce the number of features in a dataset to smaller sets while
retaining most of their useful and discriminatory information. It discusses the
standard approach to dimensionality reduction via principal component analysis
and compares it to supervised and nonlinear transformation techniques.

Chapter 6, Learning Best Practices for Model Evaluation and Hyperparameter Tuning,
discusses the do's and don'ts for estimating the performances of predictive models.
Moreover, it discusses different metrics for measuring the performance of our
models and techniques to fine-tune machine learning algorithms.

Chapter 7, Combining Different Models for Ensemble Learning, introduces you to the
different concepts of combining multiple learning algorithms effectively. It teaches
you how to build ensembles of experts to overcome the weaknesses of individual
learners, resulting in more accurate and reliable predictions.

Chapter 8, Applying Machine Learning to Sentiment Analysis, discusses the essential
steps to transform textual data into meaningful representations for machine learning
algorithms to predict the opinions of people based on their writing.

Chapter 9, Embedding a Machine Learning Model into a Web Application, continues with
the predictive model from the previous chapter and walks you through the essential
steps of developing web applications with embedded machine learning models.

Chapter 10, Predicting Continuous Target Variables with Regression Analysis, discusses
the essential techniques for modeling linear relationships between target and
response variables to make predictions on a continuous scale. After introducing
different linear models, it also talks about polynomial regression and
tree-based approaches.

Chapter 11, Working with Unlabeled Data – Clustering Analysis, shifts the focus to a
different subarea of machine learning, unsupervised learning. We apply algorithms
from three fundamental families of clustering algorithms to find groups of objects
that share a certain degree of similarity.

Chapter 12, Training Artificial Neural Networks for Image Recognition, extends the
concept of gradient-based optimization, which we first introduced in Chapter 2,
Training Machine Learning Algorithms for Classification, to build powerful, multilayer
neural networks based on the popular backpropagation algorithm.

Chapter 13, Parallelizing Neural Network Training with Theano, builds upon the
knowledge from the previous chapter to provide you with a practical guide for
training neural networks more efficiently. The focus of this chapter is on Theano, an
open source Python library that allows us to utilize multiple cores of modern GPUs.

What you need for this book

The execution of the code examples provided in this book requires an installation
of Python 3.4.3 or newer on Mac OS X, Linux, or Microsoft Windows. We will make
frequent use of Python's essential libraries for scientific computing throughout this
book, including SciPy, NumPy, scikit-learn, matplotlib, and pandas.
The first chapter will provide you with instructions and useful tips to set up your
Python environment and these core libraries. We will add additional libraries to
our repertoire and installation instructions are provided in the respective chapters:
the NLTK library for natural language processing (Chapter 8, Applying Machine
Learning to Sentiment Analysis), the Flask web framework (Chapter 9, Embedding a
Machine Learning Algorithm into a Web Application), the seaborn library for statistical
data visualization (Chapter 10, Predicting Continuous Target Variables with Regression
Analysis), and Theano for efficient neural network training on graphical processing
units (Chapter 13, Parallelizing Neural Network Training with Theano).

Who this book is for

If you want to find out how to use Python to start answering critical questions
of your data, pick up Python Machine Learning—whether you want to start from
scratch or want to extend your data science knowledge, this is an essential and
unmissable resource.

Conventions

In this book, you will find a number of text styles that distinguish between different
kinds of information. Here are some examples of these styles and an explanation of
their meaning.

Code words in text, database table names, folder names, filenames, file extensions,
pathnames, dummy URLs, user input, and Twitter handles are shown as follows:
"And already installed packages can be updated via the --upgrade flag."

A block of code is set as follows:

>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> y = df.iloc[0:100, 4].values
>>> y = np.where(y == 'Iris-setosa', -1, 1)
>>> X = df.iloc[0:100, [0, 2]].values
>>> plt.scatter(X[:50, 0], X[:50, 1],
... color='red', marker='x', label='setosa')
>>> plt.scatter(X[50:100, 0], X[50:100, 1],
... color='blue', marker='o', label='versicolor')
>>> plt.xlabel('sepal length')
>>> plt.ylabel('petal length')
>>> plt.legend(loc='upper left')
>>> plt.show()

Any command-line input or output is written as follows:

> dot -Tpng tree.dot -o tree.png

New terms and important words are shown in bold. Words that you see on the
screen, for example, in menus or dialog boxes, appear in the text like this: "After we
click on the Dashboard button in the top-right corner, we have access to the control
panel shown at the top of the page."

Reader feedback

Feedback from our readers is always welcome. Let us know what you think about
this book—what you liked or disliked. Reader feedback is important for us as it helps
us develop titles that you will really get the most out of.
To send us general feedback, simply e-mail feedback@packtpub.com, and mention
the book's title in the subject of your message.
If there is a topic that you have expertise in and you are interested in either writing
or contributing to a book, see our author guide at www.packtpub.com/authors.

Customer support

Now that you are the proud owner of a Packt book, we have a number of things to
help you to get the most from your purchase.
Downloading the example code
You can download the example code files from your account at http://www.
packtpub.com for all the Packt Publishing books you have purchased. If you
purchased this book elsewhere, you can visit http://www.packtpub.com/support
and register to have the files e-mailed directly to you.

Errata

Although we have taken every care to ensure the accuracy of our content, mistakes
do happen. If you find a mistake in one of our books—maybe a mistake in the text or
the code—we would be grateful if you could report this to us. By doing so, you can
save other readers from frustration and help us improve subsequent versions of this
book. If you find any errata, please report them by visiting http://www.packtpub.
com/submit-errata, selecting your book, clicking on the Errata Submission Form
link, and entering the details of your errata. Once your errata are verified, your
submission will be accepted and the errata will be uploaded to our website or
added to any list of existing errata under the Errata section of that title.
To view the previously submitted errata, go to https://www.packtpub.com/books/
content/support and enter the name of the book in the search field. The required
information will appear under the Errata section.

Piracy

Piracy of copyrighted material on the Internet is an ongoing problem across all
media. At Packt, we take the protection of our copyright and licenses very seriously.
If you come across any illegal copies of our works in any form on the Internet, please
provide us with the location address or website name immediately so that we can
pursue a remedy.
Please contact us at copyright@packtpub.com with a link to the suspected
pirated material.
We appreciate your help in protecting our authors and our ability to bring you
valuable content.

Questions

If you have a problem with any aspect of this book, you can contact us at
questions@packtpub.com, and we will do our best to address the problem.

