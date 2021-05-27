# Session 4 - RNN & LSTM

## Objective

- Refer to the file we wrote in the [class](https://colab.research.google.com/drive/1-xwX32O0WYOqcCROJnnJiSdzScPCudAM?usp=sharing): Rewrite this code, but this time remove the RNN and add 2 LSTM layers.

- Refer to this [file](https://colab.research.google.com/drive/12Pciev6dvYBJ7KxwSHruG-XMwcoj0SfJ). Solve the quiz.

## Solution

### Rerun the solution

The rnn solution of Sentiment Analysis. Accuracy is not as good as LSTM and have poor performance.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/pankaj90382/END-1.0/blob/main/S4/END2%20Session%204.ipynb)

### LSTM Rewrite
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/pankaj90382/END-1.0/blob/main/S4/END2_LSTM.ipynb)


To use an LSTM instead of the standard RNN, we use nn.LSTM instead of nn.RNN. Also, note that the LSTM returns the output and a tuple of the final hidden state and the final cell state, whereas the standard RNN only returned the output and final hidden state.

As the final hidden state of our LSTM has both a forward and a backward component, which will be concatenated together, the size of the input to the nn.Linear layer is twice that of the hidden dimension size.

Implementing bidirectionality and adding additional layers are done by passing values for the num_layers and bidirectional arguments for the RNN/LSTM.

Dropout is implemented by initializing an nn.Dropout layer (the argument is the probability of dropping out each neuron) and using it within the forward method after each layer we want to apply dropout to. Note: never use dropout on the input or output layers (text or fc in this case), you only ever want to use dropout on intermediate layers. The LSTM has a dropout argument which adds dropout on the connections between hidden states in one layer to hidden states in the next layer.

As we are passing the lengths of our sentences to be able to use packed padded sequences, we have to add a second argument, text_lengths, to forward.

Before we pass our embeddings to the RNN, we need to pack them, which we do with nn.utils.rnn.packed_padded_sequence. This will cause our RNN to only process the non-padded elements of our sequence. The RNN will then return packed_output (a packed sequence) as well as the hidden and cell states (both of which are tensors). Without packed padded sequences, hidden and cell are tensors from the last element in the sequence, which will most probably be a pad token, however when using packed padded sequences they are both from the last non-padded element in the sequence. Note that the lengths argument of packed_padded_sequence must be a CPU tensor so we explicitly make it one by using .to('cpu').

We then unpack the output sequence, with nn.utils.rnn.pad_packed_sequence, to transform it from a packed sequence to a tensor. The elements of output from padding tokens will be zero tensors (tensors where every element is zero). Usually, we only have to unpack output if we are going to use it later on in the model. Although we aren't in this case, we still unpack the sequence just to show how it is done.

The final hidden state, hidden, has a shape of [num layers * num directions, batch size, hid dim]. These are ordered: [forward_layer_0, backward_layer_0, forward_layer_1, backward_layer 1, ..., forward_layer_n, backward_layer n]. As we want the final (top) layer forward and backward hidden states, we get the top two hidden layers from the first dimension, hidden[-2,:,:] and hidden[-1,:,:], and concatenate them together before passing them to the linear layer (after applying dropout).

```python
class LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx )
        
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional)
        
        self.fc = nn.Linear(hidden_dim * 2 , output_dim)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):

        #text = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(text))
        
        #embedded = [sent len, batch size, emb dim]
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))

        output, (hidden, cell) = self.LSTM(packed_embedded)


        
        #output = [sent len, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        
        #assert torch.equal(output[-1,:,:], hidden.squeeze(0))

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        
        return self.fc(hidden)
```

The Invocation of LSTM in python

```python
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = LSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)
```

#### Training Logs

```
Epoch: 01 | Epoch Time: 0m 30s
	Train Loss: 0.678 | Train Acc: 56.50%
	 Val. Loss: 0.640 |  Val. Acc: 63.32%
Epoch: 02 | Epoch Time: 0m 31s
	Train Loss: 0.630 | Train Acc: 64.74%
	 Val. Loss: 0.564 |  Val. Acc: 71.35%
Epoch: 03 | Epoch Time: 0m 32s
	Train Loss: 0.585 | Train Acc: 69.06%
	 Val. Loss: 0.601 |  Val. Acc: 70.11%
Epoch: 04 | Epoch Time: 0m 33s
	Train Loss: 0.522 | Train Acc: 74.15%
	 Val. Loss: 0.620 |  Val. Acc: 66.24%
Epoch: 05 | Epoch Time: 0m 34s
	Train Loss: 0.514 | Train Acc: 74.31%
	 Val. Loss: 0.427 |  Val. Acc: 80.31%
Epoch: 06 | Epoch Time: 0m 34s
	Train Loss: 0.430 | Train Acc: 80.09%
	 Val. Loss: 0.404 |  Val. Acc: 82.40%
Epoch: 07 | Epoch Time: 0m 34s
	Train Loss: 0.386 | Train Acc: 83.29%
	 Val. Loss: 0.601 |  Val. Acc: 75.61%
Epoch: 08 | Epoch Time: 0m 34s
	Train Loss: 0.358 | Train Acc: 84.70%
	 Val. Loss: 0.442 |  Val. Acc: 81.83%
Epoch: 09 | Epoch Time: 0m 34s
	Train Loss: 0.350 | Train Acc: 85.17%
	 Val. Loss: 0.328 |  Val. Acc: 86.31%
Epoch: 10 | Epoch Time: 0m 34s
	Train Loss: 0.303 | Train Acc: 87.35%
	 Val. Loss: 0.421 |  Val. Acc: 82.32%
Epoch: 11 | Epoch Time: 0m 34s
	Train Loss: 0.293 | Train Acc: 88.00%
	 Val. Loss: 0.330 |  Val. Acc: 86.33%
Epoch: 12 | Epoch Time: 0m 34s
	Train Loss: 0.263 | Train Acc: 89.38%
	 Val. Loss: 0.389 |  Val. Acc: 85.18%
Epoch: 13 | Epoch Time: 0m 34s
	Train Loss: 0.255 | Train Acc: 89.89%
	 Val. Loss: 0.334 |  Val. Acc: 86.65%
Epoch: 14 | Epoch Time: 0m 34s
	Train Loss: 0.242 | Train Acc: 90.06%
	 Val. Loss: 0.333 |  Val. Acc: 88.29%
Epoch: 15 | Epoch Time: 0m 34s
	Train Loss: 0.233 | Train Acc: 90.80%
	 Val. Loss: 0.363 |  Val. Acc: 87.17%
Epoch: 16 | Epoch Time: 0m 34s
	Train Loss: 0.219 | Train Acc: 91.08%
	 Val. Loss: 0.303 |  Val. Acc: 89.23%
Epoch: 17 | Epoch Time: 0m 34s
	Train Loss: 0.211 | Train Acc: 91.54%
	 Val. Loss: 0.369 |  Val. Acc: 87.06%
Epoch: 18 | Epoch Time: 0m 34s
	Train Loss: 0.196 | Train Acc: 92.32%
	 Val. Loss: 0.317 |  Val. Acc: 89.27%
Epoch: 19 | Epoch Time: 0m 34s
	Train Loss: 0.188 | Train Acc: 92.69%
	 Val. Loss: 0.303 |  Val. Acc: 88.67%
Epoch: 20 | Epoch Time: 0m 34s
	Train Loss: 0.171 | Train Acc: 93.40%
	 Val. Loss: 0.350 |  Val. Acc: 88.22%
Epoch: 21 | Epoch Time: 0m 34s
	Train Loss: 0.161 | Train Acc: 93.59%
	 Val. Loss: 0.343 |  Val. Acc: 89.22%
Epoch: 22 | Epoch Time: 0m 34s
	Train Loss: 0.147 | Train Acc: 94.44%
	 Val. Loss: 0.363 |  Val. Acc: 88.37%
Epoch: 23 | Epoch Time: 0m 34s
	Train Loss: 0.145 | Train Acc: 94.45%
	 Val. Loss: 0.437 |  Val. Acc: 86.92%
Epoch: 24 | Epoch Time: 0m 34s
	Train Loss: 0.134 | Train Acc: 95.00%
	 Val. Loss: 0.427 |  Val. Acc: 86.84%
Epoch: 25 | Epoch Time: 0m 34s
	Train Loss: 0.130 | Train Acc: 95.07%
	 Val. Loss: 0.371 |  Val. Acc: 88.50%
Epoch: 26 | Epoch Time: 0m 34s
	Train Loss: 0.118 | Train Acc: 95.47%
	 Val. Loss: 0.390 |  Val. Acc: 88.86%
Epoch: 27 | Epoch Time: 0m 34s
	Train Loss: 0.118 | Train Acc: 95.62%
	 Val. Loss: 0.362 |  Val. Acc: 89.41%
Epoch: 28 | Epoch Time: 0m 34s
	Train Loss: 0.103 | Train Acc: 96.07%
	 Val. Loss: 0.397 |  Val. Acc: 88.72%
Epoch: 29 | Epoch Time: 0m 34s
	Train Loss: 0.096 | Train Acc: 96.33%
	 Val. Loss: 0.386 |  Val. Acc: 89.02%
Epoch: 30 | Epoch Time: 0m 34s
	Train Loss: 0.101 | Train Acc: 96.25%
	 Val. Loss: 0.413 |  Val. Acc: 88.76%
```

#### Results

| Text | Preictive Probability | Label |
|-----------|-----------| ----------- |
| This film is terrible | 0.001283382996916771  | Negative |
| This film is great | 0.9960214495658875  | Postive |


### Solving the Quiz
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/pankaj90382/END-1.0/blob/main/S4/EVA%20P2S3_Q7.ipynb)


The first six quiz solved however the same question repeated in EVA Phase 2, Session 10. These are simple steps to resolve sigmoid, derivative sigmoid, tanh, derivative of tanh. Some parts
included in the LSTM forward function.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/pankaj90382/END-1.0/blob/main/S4/EVA%20P2S3.ipynb)

I have seen huge accuracy differences in the run time of colab. Even this code is not using the GPU Engine, but presence of GPU Engine in colab, After 50000 epochs, the loss cames to 27 to 33. 
While using the CPU, After 50000 Epochs, the loss is 4 to 8. 


## Refernces:

- [Text_sentiment_ngrams_tutorial](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)
- [IMDB_TorchText_Interpret](https://captum.ai/tutorials/IMDB_TorchText_Interpret)
- [Sentiment Refrence](https://github.com/bentrevett/pytorch-sentiment-analysis)
