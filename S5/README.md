# Session 5 - Stanford Sentimental Data (STT)

## Objective

1.  Look at  [this code (Links to an external site.)](https://colab.research.google.com/drive/19wZi7P0Tzq9ZxeMz5EDmzfWFBLFWe6kN?usp=sharing&pli=1&authuser=3)  above. It has additional details on "Back Translate", i.e. using Google translate to convert the sentences. It has "random_swap" function, as well as "random_delete".
2.  Use "Back Translate", "random_swap" and "random_delete" to augment the data you are training on
3.  Download the StanfordSentimentAnalysis Dataset from this  [link (Links to an external site.)](http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip)(it might be troubling to download it, so force download on chrome). Use "datasetSentences.txt" and "sentiment_labels.txt" files from the zip you just downloaded as your dataset. This dataset contains just over 10,000 pieces of Stanford data from HTML files of Rotten Tomatoes. The sentiments are rated between 1 and 25, where one is the most negative and 25 is the most positive.
4.  Train your model and achieve  **60%+ validation/test accuracy**. Upload your collab file on GitHub with readme that contains details about your assignment/word (minimum  **250 words**),  **training logs showing final validation accuracy, and outcomes for  10  example inputs from the test/validation data.**


## Solution

| Model | Model with Augmentation |
|--|--|
| [Github](https://github.com/pankaj90382/END-1.0/blob/main/S5/Stanford_Data_LSTM.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/pankaj90382/END-1.0/blob/main/S5/Stanford_Data_LSTM.ipynb)| [Github](https://github.com/pankaj90382/END-1.0/blob/main/S5/Stanford_Data_LSTM_Augmented.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/pankaj90382/END-1.0/blob/main/S5/Stanford_Data_LSTM_Augmented.ipynb)|
|[Dataset](./Dataset)|[Augmented Dataset](./Dataset_Augmented)|

To use an LSTM instead of the standard RNN, I use nn.LSTM instead of nn.RNN. Also, note that the LSTM returns the output and a tuple of the final hidden state and the final cell state, whereas the standard RNN only returned the output and final hidden state.

As we are passing the lengths of our sentences to be able to use packed padded sequences, we have to add a second argument, text_lengths, to forward.

Before I pass our embeddings to the RNN, we need to pack them, which we do with nn.utils.rnn.packed_padded_sequence. This will cause our RNN to only process the non-padded elements of our sequence. The RNN will then return packed_output (a packed sequence) as well as the hidden and cell states (both of which are tensors). Without packed padded sequences, hidden and cell are tensors from the last element in the sequence, which will most probably be a pad token, however when using packed padded sequences they are both from the last non-padded element in the sequence. Note that the lengths argument of packed_padded_sequence must be a CPU tensor so we explicitly make it one by using .to('cpu').

## Augmentation



## Refrences

- [Github Data Code](https://gist.github.com/wpm/52758adbf506fd84cff3cdc7fc109aad)
- [Understanding STT Data ](https://towardsdatascience.com/the-stanford-sentiment-treebank-sst-studying-sentiment-analysis-using-nlp-e1a4cad03065)
- [EDA Data Augmentation ](https://github.com/jasonwei20/eda_nlp)
