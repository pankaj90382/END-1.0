# Session 9 NLP Evaluation Metrics

## Objective
1.  Implement the following metrics (either on separate models or same, your choice):
    1.  Recall, Precision, and F1 Score
    2.  BLEU
    3.  Perplexity (explain whether you are using bigram, trigram, or something else, what does your PPL score represent?)
    4.  BERTScore (here are  [1 (Links to an external site.)](https://colab.research.google.com/drive/1kpL8Y_AnUUiCxFjhxSrxCsc6-sDMNb_Q)  [2 (Links to an external site.)](https://huggingface.co/metrics/bertscore)  examples)


## Solution

### Evaluation Metrics

Evalauation of machine learning model is one of the major part in building a machine learning model. 'Accuracy' is one of the most commonly used Evaluation metric in the classification problems, that is the total number of correct predictions by the total number of predictions.
Accuracy, Recall, Precision, and F1- Score are the key classification metrics.

#### Accuracy:

It is defined as percentage of total number of correct predictions to the total number of observations in the dataset. It can be easily calculated as total numer of correct predictions divided by total number of predictions.
<p align="center">
  <img alt="Accuracy" src="https://user-images.githubusercontent.com/36162708/124904184-5e34b100-e002-11eb-9649-debd7a008121.png">
</p>
Where,<br>
TP => True Positive<br>
TN => True Negative<br>
FP => False Positive<br>
FN => False Negative<br>

#### Recall:
Recall is the percentage of relevant results that are correctly classified by the model, ie,it is the ratio of samples which were predicted to belong to a class with respect to all of the samples that truly belong in the class (predicted results).
<p align="center">
  <img alt="Recall" src="https://user-images.githubusercontent.com/36162708/124906039-4827f000-e004-11eb-81a5-75e26cdf3375.png">
</p>

#### Precision
Precision is the percentage of relevant results, ie, it is the ratio of True Positives(TP) to all the positives in the dataset (actual results).
<p align="center">
  <img alt="Precision" src="https://user-images.githubusercontent.com/36162708/124906773-0fd4e180-e005-11eb-9af2-b2bb9b5a5f50.png">
</p>

#### F1 Score
F1 is the weighted average of precision and recall of the model. It gives more importance to the false positives and false negatives while not letting large numbers of true negatives influence the score. A good F1 score is when there are low false positives and low false negatives. The F1 score ranges from 0 to 1 and is considered perfect when it's 1.
<p align="center">
  <img alt="F1 Score" src="https://user-images.githubusercontent.com/36162708/125073716-023d5b80-e0da-11eb-8b19-2b94e91c8cc2.png">
</p>
<br>It can also be written as
    <p align="center">
    <img alt="F1 Score" src="https://latex.codecogs.com/gif.latex?F1%20Score%20%3D%20%5Cfrac%7B2%20*%20True%20Postive%7D%7BTrue%20Postive%20&plus;%5Cfrac%7B1%7D%7B2%7D%20*%20%28False%20Postive%20&plus;%20False%20Negative%29%7D">
    </p>

#### BLEU Score
The BLEU (BiLingual Evaluation Understudy) score is a string-matching algorithm used for evaluating the quality of text which has been translated by a model from a language. The bleu metric ranges from 0 to 1 with 0 being the lowest score and 1 the highest. The closer the score is to 1, the more overlap there is with the reference translations. A higher score is also given to sequential matching words, ie, if a string of four words match the reference translation in the same exact order, it will have a more positive impact on the BLEU score than a string of two matching words.
There's this nice interpretation of BLEU Score from [Google Cloud](https://cloud.google.com/translate/automl/docs/evaluate)
| BLEU Score | Interpretation |
|--|--|
| < 10 | Almost useless
| 10 - 19 | Hard to get the gist
| 20 - 29 | The gist is clear, but has significant grammatical errors
| 30 - 40 | Understandable to good translations
| 40 - 50 | High quality translations
| 50 - 60 | Very high quality, adequate, and fluent translations
| > 60 | Quality often better than human

BLEU first makes n-grams (basically combine n words) from the predicted sentences and compare it with the n-grams of the actual target sentences. This matching is independent of the position of the n-gram. More the number of matches, more better the model is at translating.

#### Perplexity
 Perplexity is defined as a measurement of how well a probability distribution or probability model predicts a sample. A better language model will have lower perplexity values or higher probability values for a test/valid set. It is also defined as exponential average negative log-likelihood of a sequence.<br>
  If we have a tokenized sequence X = (x0,x1,x2,...,xt) then the perplexity of X is:  
 <p align="center">
  <img alt="PPL" src="https://user-images.githubusercontent.com/36162708/125092264-ca8cde80-e0ee-11eb-9ddc-f8af1478bfa1.png">
</p>

### BERTScore

![bertscore architecture](./Architecture_BERTScore.png?raw=true)

BertScore basically addresses two common pitfalls in n-gram-based metrics. Firstly, the n-gram models fail to robustly match paraphrases which leads to performance underestimation when semantically-correct phrases are penalized because of their difference from the surface form of the reference. BertScore is a metric used for evaluating the text generated by a model. It computes a similarity score for each token in the predicted sentence with each token in the reference sentence using the contextual embeddings from the BERT model and generates scores in three common metrics- precision, recall and F1 measure.
    

### Text Classification Model and Evaluation
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/pankaj90382/END-1.0/blob/main/S9/END2_LSTM.ipynb)






### Language Translation Model and Evaluation
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/pankaj90382/END-1.0/blob/main/S9/4%20-%20Packed%20Padded%20Sequences%2C%20Masking%2C%20Inference%20and%20BLEU.ipynb)

