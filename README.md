# Word2Vec Application Tutorial

The tutorial was created as a final project for IDCE 30274 Programming in GIS taught by professor Shadrock Robests at Clark University. 

In this tutorial, we will learn how to perform basic operations on word vectors. Word vectors represent words as multidimensional continuous floating point numbers where semantically similar words are mapped to proximate points in geometric space. In simpler terms, a word vector is a row of real-valued numbers where each point captures a dimension of the word's meaning and where semantically similar words have similar vectors. While there are many Natural Language Processing (NLP) libraries in Python, such as NLTK, gensim, and spaCy, we will use spaCy in this tutorial. SpaCy is popular NLP library and it provides built-in support for word vectors. 

### What is Word2Vec?

Word2Vec is a technique for natural language processing. The word2vec algorithm uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence. As the name implies, word2vec represents each distinct word with a particular list of numbers called a vector. The vectors are chosen carefully such that a simple mathematical function (the cosine similarity between the vectors) indicates the level of semantic similarity between the words represented by those vectors.

Our goal is to learn:

* Popular Python machine learning packages (spaCy, sklearn)
* Calculating word similarity using Word2Vec model
* Word analogy analysis
* Calculating sentence similarity using Word2Vec model
* Dimension reduction techniques for high-dimensional vectors
* Visualizing Word2Vec in 2D space
* Sentiment analysis using logistic regression and Word2Vec

