
# DataHacks2022

Team Members: [Huaning Liu](https://github.com/Stevela-hn), [Alan Wang](https://github.com/alanwmy00), [Hongyi Yang](https://github.com/hoy007), [Jack Yang](https://github.com/immmjack)

## Code

[Cleaning & EDA](https://github.com/alanwmy00/DataHacks2022/blob/main/Cleaning%20and%20EDA.ipynb)

[Modelling 1](https://github.com/alanwmy00/DataHacks2022/blob/main/Modelling_1.ipynb)

[Modelling 2](https://github.com/alanwmy00/DataHacks2022/blob/main/Modelling_2.ipynb)

## Introduction

Finance does not simply consist of statistics, data and monetary things, but it involves more negotiation and communication. In this way, conversations and sentiments will be a very important part that might influence the change of monetary and financial markets. In this project, given a bunch of comments as a format of texts, we would like to develop machine learning/deep learning models to predict if the sentiment included in the comments are actually positive, neutral or negative.

## Data Preprocessing & Cleaning

Before some related visualizations and modeling start, given that the pieces of information are collected in a raw website/database, a series of data/text clearing will be necessary to ensure the quality of data. In our sentiment analysis, we want to remove the unnecessary punctuations by lambda expressions and regular expressions, and also remove “stopwords”, which is a collection of words that are “neutral” and very common/everyday use words, just so we can trim out the unnecessary characters that make our dataset too long to compute. However, some common words might play a significant role in determining the sentiment, e.g., “not,” which reverses some sentiment, so we remove these words from “stopwords.”

## Exploratory Data Analysis

The first thing we notice is a difference between the numbers of three categories, which may cause a problem in our classification. So we plot a histogram to see. As noticed from this visualization, there exists some unbalance based on the count for each label, that neutral takes the largest number of counts, and positive follows by nearly halving of the neutral ones, and negative comments as labels actually take the lowest count. This is a signal for us to rationally decide if our final prediction is intuitively reasonable to submit. Also, when building our models, related regularization or adjustments could be applied to make the model more robust.

![image](https://user-images.githubusercontent.com/27839519/162640346-9ff79ee0-7028-485d-9410-bc85305514b1.png)

On the other hand, there may also exist a problem since the top-word count in each category differs significantly.

![image](https://user-images.githubusercontent.com/27839519/162640338-efcf2912-e60e-438c-84e3-f6b28d5c480a.png)


We also wish to see what words appear the most in our training set, so we create a word cloud and plot a histogram of the value counts. They basically express the same idea, but in two different methodologies and help us draw different observations from that. Note that this is also helpful for us to check if our data is actually “clean”. First, by looking at the word cloud, we reasonably noticed the appearance of some keywords in finance such as company, profit, service and sale. But also notice there are some dominant words that seem strange to our “stereotype” to finance, including finnish and finland. This is quite interesting because this probably indicates that most of the comment focuses on the market in a special area, another hot-word EUR also validates our guesses. From the perspective of modeling, most of the words do not have a clear indication of sentiments, which might be something that needs to be specially addressed in later stages. 

![image](https://user-images.githubusercontent.com/27839519/162640371-b19ad632-8b7f-4ce8-9024-050590c02766.png)
![image](https://user-images.githubusercontent.com/27839519/162640519-6f3ddbb7-16fb-435a-bab1-d0e88a5ecbd2.png)

Lastly, we wish to see how often the top 30 words show up in each category.  In the visualization below, based on the “most popular 30 words” we created, we would like to see how the counts of labels are distributed among the top-words rather than all words. Hence, noticeably, the negative comments seem to use less popular words in the vocabulary. As neutral word counts still take the dominant proportion, we guess the reason is that neural comments usually analyze both the advantages and disadvantages of somethings, which might lead to broader use of hot words we extracted.

![image](https://user-images.githubusercontent.com/27839519/162640321-db4d7940-6359-4a6a-b0f2-fc3fd44df7f6.png)


## Feature Engineering

We apply TF-IDF (Term Frequency-Inverse Document Frequency) to extract the features from the text data. This gives the overall weight of each word in the sentence, rather than the frequency of each word in the sentence simply as in the CountVectorizer. In other words, this is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. With this large dataset, we request the maximum features to be 2000. For all extracted “features” with nonzero TF-IDF, the average is around 0.3. The maximum TF-IDF for some features is 1. 

## Modeling

First, we fit the model by using Logistic Regression and Naive Bayes models. For each trial, we fix the vector given by the TF-IDF technique. The logistic regression implementation takes a regularization parameter C, so we mainly experiment with different regularization parameters. As always, in the Logistic Regression, we set the maximum iteration number to be 10000, and the random seed to be 1. For C = 2, the training accuracy is 0.8715, and the validation accuracy is 0.6715, as we leave out 30% of training data to test in the train-test split framework. For C = 3, the training accuracy is 0.89, but the validation accuracy is 0.6639, which goes a bit lower. For C = 10, the training accuracy rises up to 0.92, but the validation accuracy decreases to 0.654. It follows that increasing the regularization parameter C can increase the training accuracy, as the model might overfit the training dataset, which leads to more inaccuracy in the validation process. In the Naive Bayes model, the default setting yields a training accuracy of 0.7798, which is not as good as in the Logistic Regression, and validation accuracy of 0.647, which is not far behind the Logistic Regression. 

Second, from the perspective of deep learning, for this sentiment analysis problem, we tried developing two major kinds of models using Tensorflow and Keras - CNN and RNN. In our customized CNN model, after many experiments, the structure of layers is settled to be embedding - conv1D - max pooling - activation - activation. Basically this is not some-pretrained models, as all pretrained models we found for either torch and tensorflow are initially designed for computer vision. Therefore we decide to custom a CNN model that is not deep (as time is not allowed). But actually, although it gives us a good training accuracy of about 91%, the validation accuracy keeps reported to be about 60%. This is obviously kind of overfitting, but by trying to add some pooling and dropout layers, it cannot change the performance a lot. But actually, this is a very good baseline model to start with. Hence we take a look at another model - RNN. For RNN attempts, we design a simple RNN-LSTM model and a bidirectional RNN model; we first get the network running and compare their initial performances. Single RNN-LSTM gives a validation accuracy of about 58%, but the bidirectional model gives a validation performance of nearly 70%. In this way, given the limited time, we would like to tune and optimize based on the latter model. It is worth mentioning that we inputted two sets of features to the deep learning models: one of which is based on the popularity of each sentence (by the vocabulary counts) and the other one is based on a index set we established with the vocabulary. The latter one turns out to be better generally. The visualization below presents a loss plot for our CNN attempt, as we noticed below it actually does not strongly converge, which indicates a potentially neutral level design of the network.

![image](https://user-images.githubusercontent.com/27839519/162641548-84f46474-f477-4690-a9a5-e0e69a6ab433.png)

In this way, from the perspective of deep learning models, we finally decide to tune for the bidirectional RNN model, and it finally gives us a validation accuracy of 72%. The visualization below gives the loss plot and accuracy plot for the bidirectional RNN: even though tuning a lot and adding a high-proportional dropout layer, overfitting still exists to some extent, but we tried our best.

![image](https://user-images.githubusercontent.com/27839519/162641553-f4338cea-2d9b-46c1-b039-1c1959a11657.png)
![image](https://user-images.githubusercontent.com/27839519/162641560-5da6b21b-914f-4224-9db0-d9bb554806d5.png)

As a summary of our attempts and model tuning overnight, the table above shows related statistics, and we decide to turn in the prediction of bidirectional RNN model

|Model Name|Train Accuracy|Validation Loss|Validation Accuracy|
|:-----:|:-----:|:-----:|:-----:|
|Logistic Regression with C = 2|0.8715|/|0.6715|
|Logistic Regression with C = 3|0.89|/|0.6639
|Logistic Regression with C = 10|0.92|/|0.65|
|Naive Bayes|0.78|/|0.647|
|CNN|0.90|1.1(CrossEntropy)|0.558|
|Single RNN-LSTM|0.88|0.8|0.61|
|Bidirectional RNN|0.91|0.1|0.71|

## Analysis & Discussion

This is a general analysis for each step of our current progress that relates each part together. From our initial data exploration and visualization, we note that there exists a kind of imbalance of data, especially for the count of labels and the count of hot-words for each label. This might be a reason that leads to our model overfitting cases later. Also, the stopwords set is a very important set that decides the performance of the model. After we first processed the data, the accuracy for most of our models were just above 55%, which is not good at all. We then noticed that some of the key words in sentiment analysis, including “not” and “against” are mistakenly cleaned during our data preprocessing step. So we added them back and the model turned out to be better, which side-reflects the importance of such words in the data. Lastly, we found that the deep learning model in this case does not overperform probabilistic or linear models in the sklearn library too much. There are two potential reasons that lead to this phenomenon: the scale of data is not that large and the features we extracted for deep learning models is not enough for further boosting, which in this way will not lead to a deeper model; and some models from the library are partially self-tuned, which will optimize their performances here in this time-limited project.

## Conclusion & Future Development

Generally speaking, after many attempts on logistics regression, Naive Bayesian model, CNN and RNN-LSTM, we could reach an optimal validation accuracy of about 75%, which could hopefully perform well on the testing data. From the perspective of feature engineering, most of the processing techniques in our project are those involved in the college courses, including n-grams, bag-of-words and word2vectors. For future development about this, we might do some research for better techniques that promote the performance of models. Also, from the perspective of modeling, we noticed that there exists some overfitting cases for the model. We cannot address all of them given the limited time; but this could be further developed to ensure the performance on some unseen data. A team member’s current research interest, domain adaptation, might be a good way out.
