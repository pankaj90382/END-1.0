# Session 7 A-1 - Stanford Sentimental Data (STT)

## Objective

1. ONLY use datasetSentences.txt. (no augmentation required)
2. Your dataset must have around 12k examples.
3. Split Dataset into 70/30 Train and Test (no validation)
4. Convert floating-point labels into 5 classes (0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0) 

## Solution
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/pankaj90382/END-1.0/blob/main/S7/A-1/Stanford_Data_LSTM_Augmented.ipynb)

To use an LSTM instead of the standard RNN, I use nn.LSTM instead of nn.RNN. Also, note that the LSTM returns the output and a tuple of the final hidden state and the final cell state, whereas the standard RNN only returned the output and final hidden state.

As we are passing the lengths of our sentences to be able to use packed padded sequences, we have to add a second argument, text_lengths, to forward.

Before I pass our embeddings to the RNN, we need to pack them, which we do with nn.utils.rnn.packed_padded_sequence. This will cause our RNN to only process the non-padded elements of our sequence. The RNN will then return packed_output (a packed sequence) as well as the hidden and cell states (both of which are tensors). Without packed padded sequences, hidden and cell are tensors from the last element in the sequence, which will most probably be a pad token, however when using packed padded sequences they are both from the last non-padded element in the sequence. Note that the lengths argument of packed_padded_sequence must be a CPU tensor so we explicitly make it one by using .to('cpu').

### Cleanning the Dataset for better Accuracy

The proper cleanning the Dataset helps to join the setentences with phrases more accurately.

```python
def clean_data(x):
  char_dict = {
          '-LRB-' : '(',
          '-RRB-' : ')',
          '\xa0' : ' ',
          '\xc2' : '',
          '\xc3\x83\xc2\xa0' : 'a',
          'à' : 'a',
          'Â' : '',
          'â' : 'a',
          'ã' : 'a',
          'Ã¡' : 'a',
          'Ã¢' : 'a',
          'Ã£' : 'a',
          'Ã¦' : 'ae',
          'Ã§' : 'c',
          'Ã¨' : 'e',
          'Ã©' : 'e',
          'Ã­' : 'i',
          'Ã¯' : 'i',
          'Ã±' : 'n',
          'Ã³' : 'o',
          'Ã´' : 'o',
          'Ã¶' : 'o',
          'Ã»' : 'u',
          'Ã¼' : 'u',
          'æ' : 'ae',
          'ç' : 'c',
          'è' : 'e',
          'é' : 'e',
          'í' : 'i',
          'ï' : 'i',
          'ñ' : 'n',
          'ó' : 'o',
          'ô' : 'o',
          'ö' : 'o',
          'û' : 'u',
          'ü' : 'u'
      }
  for keys in char_dict.keys():
    x = x.replace(keys, char_dict[keys])
```

```python
def get_phrase_sentiments(base_directory):
    def group_labels(label):
        if label in ["very negative", "negative"]:
            return "negative"
        elif label in ["positive", "very positive"]:
            return "positive"
        else:
            return "neutral"

    dictionary = pandas.read_csv(os.path.join(base_directory, "dictionary.txt"), sep="|")
    dictionary.columns = ["phrase", "id"]
    dictionary = dictionary.set_index("id")

    sentiment_labels = pandas.read_csv(os.path.join(base_directory, "sentiment_labels.txt"), sep="|")
    sentiment_labels.columns = ["id", "sentiment"]
    sentiment_labels = sentiment_labels.set_index("id")

    phrase_sentiments = dictionary.join(sentiment_labels)

    phrase_sentiments["fine"] = pandas.cut(phrase_sentiments.sentiment, [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                           include_lowest=True,
                                           labels=["very negative", "negative", "neutral", "positive", "very positive"])
    phrase_sentiments["coarse"] = phrase_sentiments.fine.apply(group_labels)
    return phrase_sentiments


def get_sentence_partitions(base_directory):
    sentences = pandas.read_csv(os.path.join(base_directory, "datasetSentences.txt"), index_col="sentence_index",
                                sep="\t")
    splits = pandas.read_csv(os.path.join(base_directory, "datasetSplit.txt"), index_col="sentence_index")
    return sentences.join(splits)


def partition(base_directory):
    phrase_sentiments = get_phrase_sentiments(base_directory).reset_index(level=0)
    sentence_partitions = get_sentence_partitions(base_directory)
    # noinspection PyUnresolvedReferences
    phrase_sentiments['phrase'] = phrase_sentiments['phrase'].apply(lambda x : clean_data(x))
    sentence_partitions['sentence'] = sentence_partitions['sentence'].apply(lambda x : clean_data(x))
    data = pandas.merge(sentence_partitions, phrase_sentiments, right_on="phrase", left_on="sentence", how='left')
    data["splitset_label"] = data["splitset_label"].fillna(1).astype(int)
    data["sentence"] = data["sentence"].str.replace(r"\s('s|'d|'re|'ll|'m|'ve|n't)\b", lambda m: m.group(1))
    return data.groupby("splitset_label")
    
base_directory, output_directory = '/content/stanfordSentimentTreebank','/content/Dataset/';
os.makedirs(output_directory, exist_ok=True)
for splitset, partition in partition(base_directory):
    split_name = {1: "train", 2: "test", 3: "dev"}[splitset]
    filename = os.path.join(output_directory, "stanford-sentiment-treebank.%s.csv" % split_name)
    del partition["splitset_label"]
    partition.to_csv(filename)
```

```python
print("The Total null values in Train Data:- ",train_data['fine'].isnull().sum())
print("The Total null values in Test Data:- ",test_data['fine'].isnull().sum())
print("The Total null values in Dev Data:- ",dev_data['fine'].isnull().sum())

The Total null values in Train Data:-  3
The Total null values in Test Data:-  0
The Total null values in Dev Data:-  1
```

```python
train_data.dropna(subset=['fine'], inplace=True)
dev_data.dropna(subset=['fine'], inplace=True)
```

```python
print("The Total null values in Train Data:- ",train_data['fine'].isnull().sum())
print("The Total null values in Test Data:- ",test_data['fine'].isnull().sum())
print("The Total null values in Dev Data:- ",dev_data['fine'].isnull().sum())

The Total null values in Train Data:-  0
The Total null values in Test Data:-  0
The Total null values in Dev Data:-  0
```

```python
train_data = train_data.append(test_data, ignore_index=True)
train_data = train_data.append(dev_data, ignore_index=True)
```

```python
import torch.nn as nn
import torch.nn.functional as F

class classifier(nn.Module):
    
    # Define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout, pad_idx):
        
        super().__init__()          
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        # LSTM layer
        self.encoder = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           dropout=dropout,
                           batch_first=True,
                           bidirectional=True)
        # try using nn.GRU or nn.RNN here and compare their performances
        # try bidirectional and compare their performances
        self.projection = nn.Sequential(nn.Linear(2 * hidden_dim,hidden_dim),nn.BatchNorm1d(hidden_dim),nn.ReLU(), nn.Dropout(dropout)) 
        # Dense layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text, text_lengths):
        
        # text = [batch size, sent_length]
        embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb dim]
      
        # packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True)
        
        packed_output, (hidden, cell) = self.encoder(packed_embedded)
        #hidden = [batch size, num layers * num directions,hid dim]
        #cell = [batch size, num layers * num directions,hid dim]

        projection = self.projection(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        # Hidden = [batch size, hid dim * num directions]
        dense_outputs = self.fc(projection)   
        
        # Final activation function softmax
        output = F.softmax(dense_outputs, dim=1)
            
        return output
  ```
  
  ```
  Train Loss: 1.579 | Train Acc: 28.53%
	 Test Loss: 1.561 |  Test Acc: 31.62% 

	Train Loss: 1.543 | Train Acc: 34.01%
	 Test Loss: 1.545 |  Test Acc: 33.38% 

	Train Loss: 1.502 | Train Acc: 38.36%
	 Test Loss: 1.540 |  Test Acc: 33.92% 

	Train Loss: 1.463 | Train Acc: 42.87%
	 Test Loss: 1.525 |  Test Acc: 36.04% 

	Train Loss: 1.420 | Train Acc: 48.17%
	 Test Loss: 1.528 |  Test Acc: 34.92% 

	Train Loss: 1.375 | Train Acc: 52.56%
	 Test Loss: 1.530 |  Test Acc: 35.00% 

	Train Loss: 1.341 | Train Acc: 56.16%
	 Test Loss: 1.538 |  Test Acc: 34.46% 

	Train Loss: 1.303 | Train Acc: 60.01%
	 Test Loss: 1.527 |  Test Acc: 35.58% 

	Train Loss: 1.277 | Train Acc: 62.69%
	 Test Loss: 1.529 |  Test Acc: 35.75% 

	Train Loss: 1.257 | Train Acc: 64.65%
	 Test Loss: 1.536 |  Test Acc: 34.83% 

	Train Loss: 1.233 | Train Acc: 66.84%
	 Test Loss: 1.531 |  Test Acc: 35.83% 

	Train Loss: 1.216 | Train Acc: 68.75%
	 Test Loss: 1.541 |  Test Acc: 34.71% 

	Train Loss: 1.207 | Train Acc: 69.44%
	 Test Loss: 1.549 |  Test Acc: 33.88% 

	Train Loss: 1.189 | Train Acc: 71.31%
	 Test Loss: 1.540 |  Test Acc: 35.00% 

	Train Loss: 1.176 | Train Acc: 72.81%
	 Test Loss: 1.554 |  Test Acc: 33.46% 

	Train Loss: 1.170 | Train Acc: 73.15%
	 Test Loss: 1.555 |  Test Acc: 33.38% 

	Train Loss: 1.157 | Train Acc: 74.73%
	 Test Loss: 1.550 |  Test Acc: 33.96% 

	Train Loss: 1.141 | Train Acc: 76.57%
	 Test Loss: 1.545 |  Test Acc: 34.29% 

	Train Loss: 1.131 | Train Acc: 77.51%
	 Test Loss: 1.558 |  Test Acc: 32.83% 

	Train Loss: 1.115 | Train Acc: 79.24%
	 Test Loss: 1.561 |  Test Acc: 32.79% 

  ```
  
  ```
  	Unnamed: 0	sentence	id	phrase	sentiment	fine	coarse	Predicted_Label	Flag
2	63	Singer\/composer Bryan Adams contributes a sle...	225801.0	Singer\/composer Bryan Adams contributes a sle...	0.62500	positive	positive	positive	1
3	64	You'd think by now America would have had enou...	14646.0	You 'd think by now America would have had eno...	0.50000	neutral	neutral	neutral	1
4	65	Yet the act is still charming here .	14644.0	Yet the act is still charming here .	0.72222	positive	positive	positive	1
7	74	Part of the charm of Satin Rouge is that it av...	225402.0	Part of the charm of Satin Rouge is that it av...	0.72222	positive	positive	positive	1
14	138	Still , this flick is fun , and host to some t...	225973.0	Still , this flick is fun , and host to some t...	0.81944	very positive	positive	very positive	1
...	...	...	...	...	...	...	...	...	...
11848	7901	But it could have been worse .	222770.0	But it could have been worse .	0.36111	negative	negative	negative	1
11849	7902	Some of their jokes work , but most fail miser...	148746.0	Some of their jokes work , but most fail miser...	0.20833	negative	negative	negative	1
11852	7905	... Designed to provide a mix of smiles and te...	221766.0	... Designed to provide a mix of smiles and te...	0.22222	negative	negative	negative	1
11853	7906	it seems to me the film is about the art of ri...	163906.0	it seems to me the film is about the art of ri...	0.29167	negative	negative	negative	1
11857	7910	Schaeffer has to find some hook on which to ha...	148419.0	Schaeffer has to find some hook on which to ha...	0.27778	negative	negative	negative	1
  ```
  
  ```
  0	0	The Rock is destined to be the 21st Century's ...	226166.0	The Rock is destined to be the 21st Century 's...	0.69444	positive	positive	negative	0
1	1	The gorgeously elaborate continuation of `` Th...	226300.0	The gorgeously elaborate continuation of `` Th...	0.83333	very positive	positive	negative	0
5	66	Whether or not you're enlightened by any of De...	227114.0	Whether or not you 're enlightened by any of D...	0.83333	very positive	positive	positive	0
6	70	Just the labour involved in creating the layer...	224508.0	Just the labour involved in creating the layer...	0.87500	very positive	positive	positive	0
8	84	a screenplay more ingeniously constructed than...	228134.0	a screenplay more ingeniously constructed than...	0.83333	very positive	positive	negative	0
...	...	...	...	...	...	...	...	...	...
11850	7903	Even horror fans will most likely not find wha...	145161.0	Even horror fans will most likely not find wha...	0.12500	very negative	negative	positive	0
11851	7904	comes off like a rejected ABC Afterschool Spec...	229921.0	comes off like a rejected ABC Afterschool Spec...	0.16667	very negative	negative	negative	0
11854	7907	It's just disappointingly superficial -- a mov...	146522.0	It 's just disappointingly superficial -- a mo...	0.33333	negative	negative	positive	0
11855	7908	The title not only describes its main characte...	149944.0	The title not only describes its main characte...	0.23611	negative	negative	positive	0
11856	7909	Sometimes it feels as if it might have been ma...	148760.0	Sometimes it feels as if it might have been ma...	0.44444	neutral	neutral	negative	0
  ```
