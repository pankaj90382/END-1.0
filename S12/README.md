# Session 12 Transformer from Scratch

## Objective

Take the [repo](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/a2ee9271b5280be6994660c7982d0f44c67c3b63/ML/Pytorch/more_advanced/transformer_from_scratch/transformer_from_scratch.py), and make it work with any dataset. 

## Solution

### Dataset - Multi30K

The data consists of a set of thousands of **French to English translation** pairs. Each word in both the the languages will be represented as a one-hot vector. This process is handled by the Lang class. The data is normalized wherein it is transformed to lowercase and converted from unicode to ASCII. All non-letter characters are also omitted as part of the normalization process. Normalization is done to define the data in a standard form so as to reduce randomness and increase efficiency. Once the normalization process is completed, we reduce the size of the available input data using two parameters- the length of the sentence (maximum of 10 words) and certain prefixes found in the English language. At the end of this process, we have a standardised limited dataset of English to French pairs.

| | Transforemers from Scratch | Pytorch Transformer |
|--|--|--|
| Colab Code| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/pankaj90382/END-1.0/blob/main/S12/translation_transformer_from_scratch.ipynb) |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/pankaj90382/END-1.0/blob/main/S12/translation_transformer.ipynb) |

## Results

### Transforemers from Scratch

```
Epoch: 1, Train loss: 5.496, Val loss: 4.562, Epoch time = 47.704s
Epoch: 2, Train loss: 4.276, Val loss: 3.952, Epoch time = 49.146s
Epoch: 3, Train loss: 3.850, Val loss: 3.683, Epoch time = 48.867s
Epoch: 4, Train loss: 3.590, Val loss: 3.490, Epoch time = 49.145s
Epoch: 5, Train loss: 3.379, Val loss: 3.305, Epoch time = 48.945s
Epoch: 6, Train loss: 3.175, Val loss: 3.149, Epoch time = 49.076s
Epoch: 7, Train loss: 2.992, Val loss: 2.975, Epoch time = 48.966s
Epoch: 8, Train loss: 2.817, Val loss: 2.819, Epoch time = 49.118s
Epoch: 9, Train loss: 2.656, Val loss: 2.684, Epoch time = 49.009s
Epoch: 10, Train loss: 2.509, Val loss: 2.562, Epoch time = 48.945s
Epoch: 11, Train loss: 2.381, Val loss: 2.466, Epoch time = 49.049s
Epoch: 12, Train loss: 2.265, Val loss: 2.381, Epoch time = 49.079s
Epoch: 13, Train loss: 2.154, Val loss: 2.308, Epoch time = 49.017s
Epoch: 14, Train loss: 2.059, Val loss: 2.246, Epoch time = 49.173s
Epoch: 15, Train loss: 1.973, Val loss: 2.206, Epoch time = 49.284s
Epoch: 16, Train loss: 1.896, Val loss: 2.159, Epoch time = 49.112s
Epoch: 17, Train loss: 1.815, Val loss: 2.141, Epoch time = 49.069s
Epoch: 18, Train loss: 1.744, Val loss: 2.106, Epoch time = 48.967s
```

### Pytorch Transformer

```
Epoch: 1, Train loss: 5.321, Val loss: 4.119, Epoch time = 42.021s
Epoch: 2, Train loss: 3.768, Val loss: 3.334, Epoch time = 44.833s
Epoch: 3, Train loss: 3.162, Val loss: 2.904, Epoch time = 43.698s
Epoch: 4, Train loss: 2.771, Val loss: 2.625, Epoch time = 44.413s
Epoch: 5, Train loss: 2.481, Val loss: 2.452, Epoch time = 44.293s
Epoch: 6, Train loss: 2.251, Val loss: 2.309, Epoch time = 44.190s
Epoch: 7, Train loss: 2.055, Val loss: 2.209, Epoch time = 44.400s
Epoch: 8, Train loss: 1.896, Val loss: 2.131, Epoch time = 44.337s
Epoch: 9, Train loss: 1.754, Val loss: 2.068, Epoch time = 44.457s
Epoch: 10, Train loss: 1.627, Val loss: 2.023, Epoch time = 44.301s
Epoch: 11, Train loss: 1.517, Val loss: 1.995, Epoch time = 44.263s
Epoch: 12, Train loss: 1.419, Val loss: 1.960, Epoch time = 44.358s
Epoch: 13, Train loss: 1.334, Val loss: 1.946, Epoch time = 44.452s
Epoch: 14, Train loss: 1.256, Val loss: 1.943, Epoch time = 44.272s
Epoch: 15, Train loss: 1.180, Val loss: 1.944, Epoch time = 44.567s
Epoch: 16, Train loss: 1.106, Val loss: 1.920, Epoch time = 44.209s
Epoch: 17, Train loss: 1.038, Val loss: 1.916, Epoch time = 44.514s
Epoch: 18, Train loss: 0.978, Val loss: 1.899, Epoch time = 44.487s
```

### Examples

| Input | Transforemers from Scratch | Pytorch Transformer |
|--|--|--|
| Eine Gruppe von Menschen steht vor einem Iglu .| A group of people are gathered in front of a house . |  A group of people stand in front of an office . |
