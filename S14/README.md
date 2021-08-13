# Session 14 BERT AND BART (Transformers)

## Assignment

1.  Train BERT using the code mentioned  [here (Links to an external site.)](https://drive.google.com/file/d/1Zp2_Uka8oGDYsSe5ELk-xz6wIX8OIkB7/view?usp=sharing)  on the Squad Dataset for 20% overall samples (1/5 Epochs). Show results on 5 samples.
2.  Reproductive  [these (Links to an external site.)](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)  results, and show output on 5 samples.
3.  Reproduce the training explained in this  [blog (Links to an external site.)](https://towardsdatascience.com/bart-for-paraphrasing-with-simple-transformers-7c9ea3dfdd8c). You can decide to pick fewer datasets.

## Solution

   * [BERT QA Bot on SQUAD Dataset](#task1)
   * [BERT Sentence Classification](#task2)
   * [BART Paraphrasing](#task3)

<a id="task1"></a>
### BERT QA Bot on SQUAD Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/pankaj90382/END-1.0/blob/main/S14/BERT_Tutorial_How_To_Build_a_Question_Answering_Bot.ipynb)


**BERT**

BERT is Google’s  NLP framework, and seemingly the most influential one in recent times. </br>

*"BERT stands for Bidirectional Encoder Representations from Transformers. It is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of NLP tasks.”*

BERT is basically a trained Transformer Encoder stack. Both BERT models (Large & Base) have large number of encoder layers (Transformer Blocks) - 12 for the Base version and 24 for the large version.

BERT is released in two sizes BERTBASE and BERTLARGE. The BASE model is used to measure the performance of the architecture comparable to another architecture and the LARGE model produces state-of-the-art results that were reported in the research paper.

BERT is basically an Encoder stack of transformer architecture. A transformer architecture is an encoder-decoder network that uses self-attention on the encoder side and attention on the decoder side. BERTBASE has 12 layers in the Encoder stack while BERTLARGE has 24 layers in the Encoder stack. These are more than the Transformer architecture described in the original paper (6 encoder layers). BERT architectures (BASE and LARGE) also have larger feedforward-networks (768 and 1024 hidden units respectively), and more attention heads (12 and 16 respectively) than the Transformer architecture suggested in the original paper. It contains 512 hidden units and 8 attention heads. BERTBASE contains 110M parameters while BERTLARGE has 340M parameters.

Model Input: The first input token is `[CLS]`, which stands for Classification. Just like a normal Transformer, BERT takes a sequence of words as input.
Model Outputs: Each position of the sequence outputs a vector of size `hidden_size`. For sentence classification we only focus on the first position (the `[CLS]` token position. The vector can now be used to classify into the class you chose. If you have more classes, this last layer (Classifier Network) is only changed.

As opposed to directional models, which read the text input sequentially (left-to-right or right-to-left), the Transformer encoder reads the entire sequence of words at once. Therefore it is considered bidirectional, though it would be more accurate to say that it’s non-directional.

Before feeding word sequences into BERT, 15% of the words in each sequence are replaced with a `[MASK]` token. The model then attempts to predict the original value of the masked words, based on the context provided by the other, non-masked, words in the sequence.

The BERT loss function takes into consideration only the prediction of the masked values and ignores the prediction of the non-masked words. As a consequence, the model converges slower than directional models.

In the BERT training process, the model receives pairs of sentences as input and learns to predict if the second sentence in the pair is the subsequent sentence in the original document. During training, 50% of the inputs are a pair in which the second sentence is the subsequent sentence in the original document, while in the other 50% a random sentence from the corpus is chosen as the second sentence. The assumption is that the random sentence will be disconnected from the first sentence.

**Training Logs**

![bert qa model training loss](./SQUAD%20Training.png?raw=true)

```text
**** Running training *****
  Num examples = 144262
  Num Epochs = 1
  Batch size = 16
  Total optimization steps = 1803
Epoch:   0%|          | 0/1 [00:00<?, ?it/s]
Iteration:   0%|          | 0/1803 [00:00<?, ?it/s]/usr/local/lib/python3.7/dist-packages/pytorch_transformers/optimization.py:166: UserWarning: This overload of add_ is deprecated:
	add_(Number alpha, Tensor other)
Consider using one of the following signatures instead:
	add_(Tensor other, *, Number alpha) (Triggered internally at  /pytorch/torch/csrc/utils/python_arg_parser.cpp:1025.)
  exp_avg.mul_(beta1).add_(1.0 - beta1, grad)

Iteration:   0%|          | 2/1803 [00:01<15:00,  2.00it/s]
Iteration:   0%|          | 3/1803 [00:01<18:58,  1.58it/s]
Iteration:   0%|          | 4/1803 [00:02<21:01,  1.43it/s]
Iteration:   0%|          | 5/1803 [00:03<22:09,  1.35it/s]
Iteration:   0%|          | 6/1803 [00:04<22:51,  1.31it/s]
Iteration:   0%|          | 7/1803 [00:05<23:17,  1.29it/s]
Iteration:   0%|          | 8/1803 [00:05<23:36,  1.27it/s]
Iteration:   0%|          | 9/1803 [00:06<23:47,  1.26it/s]
Iteration:   1%|          | 10/1803 [00:07<23:56,  1.25it/s]
Iteration:   1%|          | 11/1803 [00:08<24:11,  1.23it/s]
Iteration:   1%|          | 12/1803 [00:09<24:12,  1.23it/s]
Iteration:   1%|          | 13/1803 [00:09<24:23,  1.22it/s]
Iteration:   1%|          | 14/1803 [00:10<24:32,  1.21it/s]
Iteration:   1%|          | 15/1803 [00:11<24:35,  1.21it/s]
Iteration:   1%|          | 16/1803 [00:12<24:28,  1.22it/s]
Iteration:   1%|          | 17/1803 [00:13<24:21,  1.22it/s]
Iteration:   1%|          | 18/1803 [00:14<24:17,  1.22it/s]
Iteration:   1%|          | 19/1803 [00:14<24:14,  1.23it/s]
Iteration:   1%|          | 20/1803 [00:15<24:11,  1.23it/s]
Iteration:   1%|          | 21/1803 [00:16<24:18,  1.22it/s]
Iteration:   1%|          | 22/1803 [00:17<24:26,  1.21it/s]
Iteration:   1%|▏         | 23/1803 [00:18<24:30,  1.21it/s]
Iteration:   1%|▏         | 24/1803 [00:19<24:31,  1.21it/s]
Iteration:   1%|▏         | 25/1803 [00:19<24:31,  1.21it/s]
Iteration:   1%|▏         | 26/1803 [00:20<24:33,  1.21it/s]
Iteration:   1%|▏         | 27/1803 [00:21<24:37,  1.20it/s]
Iteration:   2%|▏         | 28/1803 [00:22<24:42,  1.20it/s]
Iteration:   2%|▏         | 29/1803 [00:23<24:41,  1.20it/s]
Iteration:   2%|▏         | 30/1803 [00:24<24:43,  1.20it/s]
Iteration:   2%|▏         | 31/1803 [00:24<24:45,  1.19it/s]
Iteration:   2%|▏         | 32/1803 [00:25<24:53,  1.19it/s]
Iteration:   2%|▏         | 33/1803 [00:26<24:56,  1.18it/s]
Iteration:   2%|▏         | 34/1803 [00:27<24:58,  1.18it/s]
Iteration:   2%|▏         | 35/1803 [00:28<24:59,  1.18it/s]
Iteration:   2%|▏         | 36/1803 [00:29<25:02,  1.18it/s]
Iteration:   2%|▏         | 37/1803 [00:30<25:04,  1.17it/s]
Iteration:   2%|▏         | 38/1803 [00:30<25:05,  1.17it/s]
Iteration:   2%|▏         | 39/1803 [00:31<25:10,  1.17it/s]
Iteration:   2%|▏         | 40/1803 [00:32<25:15,  1.16it/s]
Iteration:   2%|▏         | 41/1803 [00:33<25:21,  1.16it/s]
Iteration:   2%|▏         | 42/1803 [00:34<25:24,  1.16it/s]
Iteration:   2%|▏         | 43/1803 [00:35<25:26,  1.15it/s]
Iteration:   2%|▏         | 44/1803 [00:36<25:34,  1.15it/s]
Iteration:   2%|▏         | 45/1803 [00:36<25:33,  1.15it/s]
Iteration:   3%|▎         | 46/1803 [00:37<25:34,  1.15it/s]
Iteration:   3%|▎         | 47/1803 [00:38<25:33,  1.14it/s]
Iteration:   3%|▎         | 48/1803 [00:39<25:33,  1.14it/s]
Iteration:   3%|▎         | 49/1803 [00:40<25:35,  1.14it/s]
Iteration:   3%|▎         | 50/1803 [00:41<25:38,  1.14it/s]
Iteration:   3%|▎         | 51/1803 [00:42<25:45,  1.13it/s]
Iteration:   3%|▎         | 52/1803 [00:43<25:44,  1.13it/s]
Iteration:   3%|▎         | 53/1803 [00:44<25:48,  1.13it/s]
Iteration:   3%|▎         | 54/1803 [00:44<25:51,  1.13it/s]
Iteration:   3%|▎         | 55/1803 [00:45<25:59,  1.12it/s]
Iteration:   3%|▎         | 56/1803 [00:46<25:58,  1.12it/s]
Iteration:   3%|▎         | 57/1803 [00:47<26:00,  1.12it/s]
Iteration:   3%|▎         | 58/1803 [00:48<26:00,  1.12it/s]
Iteration:   3%|▎         | 59/1803 [00:49<26:00,  1.12it/s]
Iteration:   3%|▎         | 60/1803 [00:50<26:00,  1.12it/s]
Iteration:   3%|▎         | 61/1803 [00:51<26:00,  1.12it/s]
Iteration:   3%|▎         | 62/1803 [00:52<26:00,  1.12it/s]
Iteration:   3%|▎         | 63/1803 [00:52<26:01,  1.11it/s]
Iteration:   4%|▎         | 64/1803 [00:53<26:00,  1.11it/s]
Iteration:   4%|▎         | 65/1803 [00:54<25:56,  1.12it/s]
Iteration:   4%|▎         | 66/1803 [00:55<25:52,  1.12it/s]
Iteration:   4%|▎         | 67/1803 [00:56<25:50,  1.12it/s]
Iteration:   4%|▍         | 68/1803 [00:57<25:48,  1.12it/s]
Iteration:   4%|▍         | 69/1803 [00:58<25:41,  1.13it/s]
Iteration:   4%|▍         | 70/1803 [00:59<25:36,  1.13it/s]
Iteration:   4%|▍         | 71/1803 [01:00<25:35,  1.13it/s]
Iteration:   4%|▍         | 72/1803 [01:00<25:30,  1.13it/s]
Iteration:   4%|▍         | 73/1803 [01:01<25:22,  1.14it/s]
Iteration:   4%|▍         | 74/1803 [01:02<25:19,  1.14it/s]
Iteration:   4%|▍         | 75/1803 [01:03<25:20,  1.14it/s]
Iteration:   4%|▍         | 76/1803 [01:04<25:13,  1.14it/s]
Iteration:   4%|▍         | 77/1803 [01:05<25:09,  1.14it/s]
Iteration:   4%|▍         | 78/1803 [01:06<25:02,  1.15it/s]
Iteration:   4%|▍         | 79/1803 [01:07<24:59,  1.15it/s]
Iteration:   4%|▍         | 80/1803 [01:07<24:53,  1.15it/s]
Iteration:   4%|▍         | 81/1803 [01:08<24:52,  1.15it/s]
Iteration:   5%|▍         | 82/1803 [01:09<24:44,  1.16it/s]
Iteration:   5%|▍         | 83/1803 [01:10<24:43,  1.16it/s]
Iteration:   5%|▍         | 84/1803 [01:11<24:41,  1.16it/s]
Iteration:   5%|▍         | 85/1803 [01:12<24:38,  1.16it/s]
Iteration:   5%|▍         | 86/1803 [01:13<24:38,  1.16it/s]
Iteration:   5%|▍         | 87/1803 [01:13<24:34,  1.16it/s]
Iteration:   5%|▍         | 88/1803 [01:14<24:30,  1.17it/s]
Iteration:   5%|▍         | 89/1803 [01:15<24:27,  1.17it/s]
Iteration:   5%|▍         | 90/1803 [01:16<24:25,  1.17it/s]
Iteration:   5%|▌         | 91/1803 [01:17<24:18,  1.17it/s]
Iteration:   5%|▌         | 92/1803 [01:18<24:17,  1.17it/s]
Iteration:   5%|▌         | 93/1803 [01:19<24:18,  1.17it/s]
Iteration:   5%|▌         | 94/1803 [01:19<24:16,  1.17it/s]
Iteration:   5%|▌         | 95/1803 [01:20<24:16,  1.17it/s]
Iteration:   5%|▌         | 96/1803 [01:21<24:11,  1.18it/s]
Iteration:   5%|▌         | 97/1803 [01:22<24:10,  1.18it/s]
Iteration:   5%|▌         | 98/1803 [01:23<24:10,  1.18it/s]
Iteration:   5%|▌         | 99/1803 [01:24<24:05,  1.18it/s]
Iteration:   6%|▌         | 100/1803 [01:24<24:01,  1.18it/s]
Iteration:   6%|▌         | 101/1803 [01:25<23:56,  1.18it/s]
Iteration:   6%|▌         | 102/1803 [01:26<23:56,  1.18it/s]
Iteration:   6%|▌         | 103/1803 [01:27<23:53,  1.19it/s]
Iteration:   6%|▌         | 104/1803 [01:28<23:54,  1.18it/s]
Iteration:   6%|▌         | 105/1803 [01:29<23:52,  1.19it/s]
Iteration:   6%|▌         | 106/1803 [01:30<23:48,  1.19it/s]
Iteration:   6%|▌         | 107/1803 [01:30<23:43,  1.19it/s]
Iteration:   6%|▌         | 108/1803 [01:31<23:41,  1.19it/s]
Iteration:   6%|▌         | 109/1803 [01:32<23:44,  1.19it/s]
Iteration:   6%|▌         | 110/1803 [01:33<23:43,  1.19it/s]
Iteration:   6%|▌         | 111/1803 [01:34<23:44,  1.19it/s]
Iteration:   6%|▌         | 112/1803 [01:35<23:47,  1.18it/s]
Iteration:   6%|▋         | 113/1803 [01:35<23:41,  1.19it/s]
Iteration:   6%|▋         | 114/1803 [01:36<23:39,  1.19it/s]
Iteration:   6%|▋         | 115/1803 [01:37<23:39,  1.19it/s]
Iteration:   6%|▋         | 116/1803 [01:38<23:43,  1.18it/s]
Iteration:   6%|▋         | 117/1803 [01:39<23:39,  1.19it/s]
Iteration:   7%|▋         | 118/1803 [01:40<23:43,  1.18it/s]
Iteration:   7%|▋         | 119/1803 [01:41<23:47,  1.18it/s]
Iteration:   7%|▋         | 120/1803 [01:41<23:48,  1.18it/s]
Iteration:   7%|▋         | 121/1803 [01:42<23:48,  1.18it/s]
Iteration:   7%|▋         | 122/1803 [01:43<23:46,  1.18it/s]
Iteration:   7%|▋         | 123/1803 [01:44<23:50,  1.17it/s]
Iteration:   7%|▋         | 124/1803 [01:45<23:51,  1.17it/s]
Iteration:   7%|▋         | 125/1803 [01:46<23:59,  1.17it/s]
Iteration:   7%|▋         | 126/1803 [01:46<23:55,  1.17it/s]
Iteration:   7%|▋         | 127/1803 [01:47<23:55,  1.17it/s]
Iteration:   7%|▋         | 128/1803 [01:48<23:52,  1.17it/s]
Iteration:   7%|▋         | 129/1803 [01:49<23:52,  1.17it/s]
Iteration:   7%|▋         | 130/1803 [01:50<23:49,  1.17it/s]
Iteration:   7%|▋         | 131/1803 [01:51<23:54,  1.17it/s]
Iteration:   7%|▋         | 132/1803 [01:52<23:52,  1.17it/s]
Iteration:   7%|▋         | 133/1803 [01:52<23:52,  1.17it/s]
Iteration:   7%|▋         | 134/1803 [01:53<23:54,  1.16it/s]
Iteration:   7%|▋         | 135/1803 [01:54<23:51,  1.17it/s]
Iteration:   8%|▊         | 136/1803 [01:55<23:51,  1.16it/s]
Iteration:   8%|▊         | 137/1803 [01:56<23:54,  1.16it/s]
Iteration:   8%|▊         | 138/1803 [01:57<23:51,  1.16it/s]
Iteration:   8%|▊         | 139/1803 [01:58<23:54,  1.16it/s]
Iteration:   8%|▊         | 140/1803 [01:59<23:58,  1.16it/s]
Iteration:   8%|▊         | 141/1803 [01:59<24:00,  1.15it/s]
Iteration:   8%|▊         | 142/1803 [02:00<23:59,  1.15it/s]
Iteration:   8%|▊         | 143/1803 [02:01<23:59,  1.15it/s]
Iteration:   8%|▊         | 144/1803 [02:02<23:58,  1.15it/s]
Iteration:   8%|▊         | 145/1803 [02:03<23:57,  1.15it/s]
Iteration:   8%|▊         | 146/1803 [02:04<24:02,  1.15it/s]
Iteration:   8%|▊         | 147/1803 [02:05<24:04,  1.15it/s]
Iteration:   8%|▊         | 148/1803 [02:05<24:02,  1.15it/s]
Iteration:   8%|▊         | 149/1803 [02:06<23:59,  1.15it/s]
Iteration:   8%|▊         | 150/1803 [02:07<23:57,  1.15it/s]
Iteration:   8%|▊         | 151/1803 [02:08<24:01,  1.15it/s]
Iteration:   8%|▊         | 152/1803 [02:09<23:58,  1.15it/s]
Iteration:   8%|▊         | 153/1803 [02:10<23:58,  1.15it/s]
Iteration:   9%|▊         | 154/1803 [02:11<23:56,  1.15it/s]
Iteration:   9%|▊         | 155/1803 [02:12<23:50,  1.15it/s]
Iteration:   9%|▊         | 156/1803 [02:12<23:50,  1.15it/s]
Iteration:   9%|▊         | 157/1803 [02:13<23:52,  1.15it/s]
Iteration:   9%|▉         | 158/1803 [02:14<23:51,  1.15it/s]
Iteration:   9%|▉         | 159/1803 [02:15<23:50,  1.15it/s]
Iteration:   9%|▉         | 160/1803 [02:16<23:46,  1.15it/s]
Iteration:   9%|▉         | 161/1803 [02:17<23:45,  1.15it/s]
Iteration:   9%|▉         | 162/1803 [02:18<23:50,  1.15it/s]
Iteration:   9%|▉         | 163/1803 [02:19<23:46,  1.15it/s]
Iteration:   9%|▉         | 164/1803 [02:19<23:44,  1.15it/s]
Iteration:   9%|▉         | 165/1803 [02:20<23:40,  1.15it/s]
Iteration:   9%|▉         | 166/1803 [02:21<23:38,  1.15it/s]
Iteration:   9%|▉         | 167/1803 [02:22<23:34,  1.16it/s]
Iteration:   9%|▉         | 168/1803 [02:23<23:32,  1.16it/s]
Iteration:   9%|▉         | 169/1803 [02:24<23:29,  1.16it/s]
Iteration:   9%|▉         | 170/1803 [02:25<23:30,  1.16it/s]
Iteration:   9%|▉         | 171/1803 [02:25<23:25,  1.16it/s]
Iteration:  10%|▉         | 172/1803 [02:26<23:24,  1.16it/s]
Iteration:  10%|▉         | 173/1803 [02:27<23:22,  1.16it/s]
Iteration:  10%|▉         | 174/1803 [02:28<23:19,  1.16it/s]
Iteration:  10%|▉         | 175/1803 [02:29<23:16,  1.17it/s]
Iteration:  10%|▉         | 176/1803 [02:30<23:15,  1.17it/s]
Iteration:  10%|▉         | 177/1803 [02:31<23:16,  1.16it/s]
Iteration:  10%|▉         | 178/1803 [02:31<23:14,  1.17it/s]
Iteration:  10%|▉         | 179/1803 [02:32<23:15,  1.16it/s]
Iteration:  10%|▉         | 180/1803 [02:33<23:10,  1.17it/s]
Iteration:  10%|█         | 181/1803 [02:34<23:11,  1.17it/s]
Iteration:  10%|█         | 182/1803 [02:35<23:08,  1.17it/s]
Iteration:  10%|█         | 183/1803 [02:36<23:06,  1.17it/s]
Iteration:  10%|█         | 184/1803 [02:37<23:05,  1.17it/s]
Iteration:  10%|█         | 185/1803 [02:37<23:08,  1.17it/s]
Iteration:  10%|█         | 186/1803 [02:38<23:04,  1.17it/s]
Iteration:  10%|█         | 187/1803 [02:39<23:03,  1.17it/s]
Iteration:  10%|█         | 188/1803 [02:40<22:59,  1.17it/s]
Iteration:  10%|█         | 189/1803 [02:41<22:59,  1.17it/s]
Iteration:  11%|█         | 190/1803 [02:42<22:56,  1.17it/s]
Iteration:  11%|█         | 191/1803 [02:43<22:57,  1.17it/s]
Iteration:  11%|█         | 192/1803 [02:43<22:55,  1.17it/s]
Iteration:  11%|█         | 193/1803 [02:44<22:53,  1.17it/s]
Iteration:  11%|█         | 194/1803 [02:45<22:55,  1.17it/s]
Iteration:  11%|█         | 195/1803 [02:46<22:56,  1.17it/s]
Iteration:  11%|█         | 196/1803 [02:47<22:55,  1.17it/s]
Iteration:  11%|█         | 197/1803 [02:48<22:51,  1.17it/s]
Iteration:  11%|█         | 198/1803 [02:49<22:53,  1.17it/s]
Iteration:  11%|█         | 199/1803 [02:49<22:51,  1.17it/s]
Iteration:  11%|█         | 200/1803 [02:50<22:54,  1.17it/s]
Iteration:  11%|█         | 201/1803 [02:51<22:53,  1.17it/s]
Iteration:  11%|█         | 202/1803 [02:52<22:52,  1.17it/s]
Iteration:  11%|█▏        | 203/1803 [02:53<22:53,  1.16it/s]
Iteration:  11%|█▏        | 204/1803 [02:54<22:50,  1.17it/s]
Iteration:  11%|█▏        | 205/1803 [02:55<22:50,  1.17it/s]
Iteration:  11%|█▏        | 206/1803 [02:55<22:46,  1.17it/s]
Iteration:  11%|█▏        | 207/1803 [02:56<22:49,  1.17it/s]
Iteration:  12%|█▏        | 208/1803 [02:57<22:48,  1.17it/s]
Iteration:  12%|█▏        | 209/1803 [02:58<22:49,  1.16it/s]
Iteration:  12%|█▏        | 210/1803 [02:59<22:51,  1.16it/s]
Iteration:  12%|█▏        | 211/1803 [03:00<22:45,  1.17it/s]
Iteration:  12%|█▏        | 212/1803 [03:01<22:46,  1.16it/s]
Iteration:  12%|█▏        | 213/1803 [03:01<22:45,  1.16it/s]
Iteration:  12%|█▏        | 214/1803 [03:02<22:47,  1.16it/s]
Iteration:  12%|█▏        | 215/1803 [03:03<22:44,  1.16it/s]
Iteration:  12%|█▏        | 216/1803 [03:04<22:46,  1.16it/s]
Iteration:  12%|█▏        | 217/1803 [03:05<22:42,  1.16it/s]
Iteration:  12%|█▏        | 218/1803 [03:06<22:41,  1.16it/s]
Iteration:  12%|█▏        | 219/1803 [03:07<22:41,  1.16it/s]
Iteration:  12%|█▏        | 220/1803 [03:07<22:40,  1.16it/s]
Iteration:  12%|█▏        | 221/1803 [03:08<22:44,  1.16it/s]
Iteration:  12%|█▏        | 222/1803 [03:09<22:44,  1.16it/s]
Iteration:  12%|█▏        | 223/1803 [03:10<22:43,  1.16it/s]
Iteration:  12%|█▏        | 224/1803 [03:11<22:40,  1.16it/s]
Iteration:  12%|█▏        | 225/1803 [03:12<22:37,  1.16it/s]
Iteration:  13%|█▎        | 226/1803 [03:13<22:36,  1.16it/s]
Iteration:  13%|█▎        | 227/1803 [03:13<22:37,  1.16it/s]
Iteration:  13%|█▎        | 228/1803 [03:14<22:34,  1.16it/s]
Iteration:  13%|█▎        | 229/1803 [03:15<22:35,  1.16it/s]
Iteration:  13%|█▎        | 230/1803 [03:16<22:36,  1.16it/s]
Iteration:  13%|█▎        | 231/1803 [03:17<22:31,  1.16it/s]
Iteration:  13%|█▎        | 232/1803 [03:18<22:31,  1.16it/s]
Iteration:  13%|█▎        | 233/1803 [03:19<22:29,  1.16it/s]
Iteration:  13%|█▎        | 234/1803 [03:20<22:31,  1.16it/s]
Iteration:  13%|█▎        | 235/1803 [03:20<22:32,  1.16it/s]
Iteration:  13%|█▎        | 236/1803 [03:21<22:33,  1.16it/s]
Iteration:  13%|█▎        | 237/1803 [03:22<22:28,  1.16it/s]
Iteration:  13%|█▎        | 238/1803 [03:23<22:31,  1.16it/s]
Iteration:  13%|█▎        | 239/1803 [03:24<22:30,  1.16it/s]
Iteration:  13%|█▎        | 240/1803 [03:25<22:33,  1.16it/s]
Iteration:  13%|█▎        | 241/1803 [03:26<22:32,  1.15it/s]
Iteration:  13%|█▎        | 242/1803 [03:26<22:34,  1.15it/s]
Iteration:  13%|█▎        | 243/1803 [03:27<22:25,  1.16it/s]
Iteration:  14%|█▎        | 244/1803 [03:28<22:26,  1.16it/s]
Iteration:  14%|█▎        | 245/1803 [03:29<22:28,  1.16it/s]
Iteration:  14%|█▎        | 246/1803 [03:30<22:24,  1.16it/s]
Iteration:  14%|█▎        | 247/1803 [03:31<22:24,  1.16it/s]
Iteration:  14%|█▍        | 248/1803 [03:32<22:20,  1.16it/s]
Iteration:  14%|█▍        | 249/1803 [03:32<22:21,  1.16it/s]
Iteration:  14%|█▍        | 250/1803 [03:33<22:18,  1.16it/s]
Iteration:  14%|█▍        | 251/1803 [03:34<22:16,  1.16it/s]
Iteration:  14%|█▍        | 252/1803 [03:35<22:17,  1.16it/s]
Iteration:  14%|█▍        | 253/1803 [03:36<22:16,  1.16it/s]
Iteration:  14%|█▍        | 254/1803 [03:37<22:15,  1.16it/s]
Iteration:  14%|█▍        | 255/1803 [03:38<22:16,  1.16it/s]
Iteration:  14%|█▍        | 256/1803 [03:39<22:15,  1.16it/s]
Iteration:  14%|█▍        | 257/1803 [03:39<22:11,  1.16it/s]
Iteration:  14%|█▍        | 258/1803 [03:40<22:12,  1.16it/s]
Iteration:  14%|█▍        | 259/1803 [03:41<22:11,  1.16it/s]
Iteration:  14%|█▍        | 260/1803 [03:42<22:10,  1.16it/s]
Iteration:  14%|█▍        | 261/1803 [03:43<22:12,  1.16it/s]
Iteration:  15%|█▍        | 262/1803 [03:44<22:08,  1.16it/s]
Iteration:  15%|█▍        | 263/1803 [03:45<22:07,  1.16it/s]
Iteration:  15%|█▍        | 264/1803 [03:45<22:06,  1.16it/s]

.
.
.
.
.
.
.
.

Iteration:  55%|█████▍    | 984/1803 [14:05<11:46,  1.16it/s]
Iteration:  55%|█████▍    | 985/1803 [14:06<11:45,  1.16it/s]
Iteration:  55%|█████▍    | 986/1803 [14:06<11:43,  1.16it/s]
Iteration:  55%|█████▍    | 987/1803 [14:07<11:41,  1.16it/s]
Iteration:  55%|█████▍    | 988/1803 [14:08<11:41,  1.16it/s]
Iteration:  55%|█████▍    | 989/1803 [14:09<11:39,  1.16it/s]
Iteration:  55%|█████▍    | 990/1803 [14:10<11:37,  1.17it/s]
Iteration:  55%|█████▍    | 991/1803 [14:11<11:38,  1.16it/s]
Iteration:  55%|█████▌    | 992/1803 [14:12<11:38,  1.16it/s]
Iteration:  55%|█████▌    | 993/1803 [14:12<11:37,  1.16it/s]
Iteration:  55%|█████▌    | 994/1803 [14:13<11:37,  1.16it/s]
Iteration:  55%|█████▌    | 995/1803 [14:14<11:35,  1.16it/s]
Iteration:  55%|█████▌    | 996/1803 [14:15<11:34,  1.16it/s]
Iteration:  55%|█████▌    | 997/1803 [14:16<11:33,  1.16it/s]
Iteration:  55%|█████▌    | 998/1803 [14:17<11:32,  1.16it/s]
Iteration:  55%|█████▌    | 999/1803 [14:18<11:31,  1.16it/s]
Iteration:  55%|█████▌    | 1000/1803 [14:18<11:29,  1.16it/s]Train loss: 1.76988218998909

Iteration:  56%|█████▌    | 1001/1803 [14:21<18:24,  1.38s/it]Saving model checkpoint to checkpoint-1000

Iteration:  56%|█████▌    | 1002/1803 [14:22<16:09,  1.21s/it]
Iteration:  56%|█████▌    | 1003/1803 [14:23<14:41,  1.10s/it]
Iteration:  56%|█████▌    | 1004/1803 [14:24<13:41,  1.03s/it]
Iteration:  56%|█████▌    | 1005/1803 [14:24<13:01,  1.02it/s]
Iteration:  56%|█████▌    | 1006/1803 [14:25<12:32,  1.06it/s]
Iteration:  56%|█████▌    | 1007/1803 [14:26<12:11,  1.09it/s]
Iteration:  56%|█████▌    | 1008/1803 [14:27<11:55,  1.11it/s]
Iteration:  56%|█████▌    | 1009/1803 [14:28<11:45,  1.13it/s]
Iteration:  56%|█████▌    | 1010/1803 [14:29<11:35,  1.14it/s]
Iteration:  56%|█████▌    | 1011/1803 [14:30<11:30,  1.15it/s]
Iteration:  56%|█████▌    | 1012/1803 [14:30<11:25,  1.15it/s]
Iteration:  56%|█████▌    | 1013/1803 [14:31<11:25,  1.15it/s]
Iteration:  56%|█████▌    | 1014/1803 [14:32<11:20,  1.16it/s]
Iteration:  56%|█████▋    | 1015/1803 [14:33<11:19,  1.16it/s]
Iteration:  56%|█████▋    | 1016/1803 [14:34<11:16,  1.16it/s]
Iteration:  56%|█████▋    | 1017/1803 [14:35<11:16,  1.16it/s]
Iteration:  56%|█████▋    | 1018/1803 [14:36<11:15,  1.16it/s]
Iteration:  57%|█████▋    | 1019/1803 [14:36<11:16,  1.16it/s]
Iteration:  57%|█████▋    | 1020/1803 [14:37<11:14,  1.16it/s]
Iteration:  57%|█████▋    | 1021/1803 [14:38<11:13,  1.16it/s]
Iteration:  57%|█████▋    | 1022/1803 [14:39<11:12,  1.16it/s]
Iteration:  57%|█████▋    | 1023/1803 [14:40<11:10,  1.16it/s]
Iteration:  57%|█████▋    | 1024/1803 [14:41<11:08,  1.16it/s]
Iteration:  57%|█████▋    | 1025/1803 [14:42<11:09,  1.16it/s]
Iteration:  57%|█████▋    | 1026/1803 [14:42<11:09,  1.16it/s]
Iteration:  57%|█████▋    | 1027/1803 [14:43<11:09,  1.16it/s]
Iteration:  57%|█████▋    | 1028/1803 [14:44<11:09,  1.16it/s]
Iteration:  57%|█████▋    | 1029/1803 [14:45<11:07,  1.16it/s]
Iteration:  57%|█████▋    | 1030/1803 [14:46<11:07,  1.16it/s]
Iteration:  57%|█████▋    | 1031/1803 [14:47<11:05,  1.16it/s]
Iteration:  57%|█████▋    | 1032/1803 [14:48<11:05,  1.16it/s]
Iteration:  57%|█████▋    | 1033/1803 [14:49<11:03,  1.16it/s]

.
.
.
.
.
.
.
.
Iteration:  98%|█████████▊| 1764/1803 [25:17<00:33,  1.16it/s]
Iteration:  98%|█████████▊| 1765/1803 [25:17<00:32,  1.16it/s]
Iteration:  98%|█████████▊| 1766/1803 [25:18<00:31,  1.16it/s]
Iteration:  98%|█████████▊| 1767/1803 [25:19<00:31,  1.16it/s]
Iteration:  98%|█████████▊| 1768/1803 [25:20<00:30,  1.16it/s]
Iteration:  98%|█████████▊| 1769/1803 [25:21<00:29,  1.16it/s]
Iteration:  98%|█████████▊| 1770/1803 [25:22<00:28,  1.16it/s]
Iteration:  98%|█████████▊| 1771/1803 [25:23<00:27,  1.15it/s]
Iteration:  98%|█████████▊| 1772/1803 [25:23<00:26,  1.15it/s]
Iteration:  98%|█████████▊| 1773/1803 [25:24<00:25,  1.15it/s]
Iteration:  98%|█████████▊| 1774/1803 [25:25<00:25,  1.16it/s]
Iteration:  98%|█████████▊| 1775/1803 [25:26<00:24,  1.16it/s]
Iteration:  99%|█████████▊| 1776/1803 [25:27<00:23,  1.16it/s]
Iteration:  99%|█████████▊| 1777/1803 [25:28<00:22,  1.16it/s]
Iteration:  99%|█████████▊| 1778/1803 [25:29<00:21,  1.16it/s]
Iteration:  99%|█████████▊| 1779/1803 [25:29<00:20,  1.16it/s]
Iteration:  99%|█████████▊| 1780/1803 [25:30<00:19,  1.16it/s]
Iteration:  99%|█████████▉| 1781/1803 [25:31<00:18,  1.16it/s]
Iteration:  99%|█████████▉| 1782/1803 [25:32<00:18,  1.16it/s]
Iteration:  99%|█████████▉| 1783/1803 [25:33<00:17,  1.16it/s]
Iteration:  99%|█████████▉| 1784/1803 [25:34<00:16,  1.16it/s]
Iteration:  99%|█████████▉| 1785/1803 [25:35<00:15,  1.16it/s]
Iteration:  99%|█████████▉| 1786/1803 [25:35<00:14,  1.16it/s]
Iteration:  99%|█████████▉| 1787/1803 [25:36<00:13,  1.16it/s]
Iteration:  99%|█████████▉| 1788/1803 [25:37<00:12,  1.16it/s]
Iteration:  99%|█████████▉| 1789/1803 [25:38<00:12,  1.17it/s]
Iteration:  99%|█████████▉| 1790/1803 [25:39<00:11,  1.17it/s]
Iteration:  99%|█████████▉| 1791/1803 [25:40<00:10,  1.16it/s]
Iteration:  99%|█████████▉| 1792/1803 [25:41<00:09,  1.16it/s]
Iteration:  99%|█████████▉| 1793/1803 [25:41<00:08,  1.16it/s]
Iteration: 100%|█████████▉| 1794/1803 [25:42<00:07,  1.16it/s]
Iteration: 100%|█████████▉| 1795/1803 [25:43<00:06,  1.16it/s]
Iteration: 100%|█████████▉| 1796/1803 [25:44<00:06,  1.16it/s]
Iteration: 100%|█████████▉| 1797/1803 [25:45<00:05,  1.16it/s]
Iteration: 100%|█████████▉| 1798/1803 [25:46<00:04,  1.16it/s]
Iteration: 100%|█████████▉| 1799/1803 [25:47<00:03,  1.17it/s]
Iteration: 100%|█████████▉| 1800/1803 [25:48<00:02,  1.16it/s]
Iteration: 100%|█████████▉| 1801/1803 [25:48<00:01,  1.16it/s]
Iteration: 100%|█████████▉| 1802/1803 [25:49<00:00,  1.16it/s]
Iteration: 100%|██████████| 1803/1803 [25:50<00:00,  1.16it/s]
Epoch: 100%|██████████| 1/1 [25:50<00:00, 1550.62s/it]
```

**Sample Outputs**

```text

question       >> The immune systems of bacteria have enzymes that protect against infection by what kind of cells?
model's answer >> unicellular

question       >> Where is the lowest point of Warsaw located?
model's answer >> 75.6 metres (248.0 ft) (at the right bank of the Vistula, by the eastern border of Warsaw

question       >> What river runs alongside Jacksonville?
model's answer >> St. Johns River

question       >> What do a and b represent in a Gaussian integer expression? 
model's answer >> arbitrary integers

question       >> What term resulted from Dioscorides' book?
model's answer >> materia medica

question       >> What is another name for State Route 168?
model's answer >> the Sierra Freeway

question       >> Does packet switching charge a fee when no data is transferred?
model's answer >> fee per unit of connection time

question       >> What months out of the year is Woodward Park open?
model's answer >> April through October

question       >> Which University ranks having the 9th most number of volumes in the US?
model's answer >> University of Chicago

question       >> What has been the main reason for the shift to the view that income inequality harms growth?
model's answer >> increasing importance of human capital in development

question       >> Strictly speaking who was included in DATANET 1
model's answer >> Dutch PTT Telecom

question       >> What color is the sputum of those suffering from septicemic plague sufferers?
model's answer >> bright red

question       >> What is  DECnet
model's answer >> a suite of network protocols

question       >> Where did the Normans and Byzantines sign the peace treaty?
model's answer >> Deabolis

question       >> What type of engines became popular for power generation after piston steam engines?
model's answer >> Reciprocating

```

---

<a id="task2"></a>
 ### BERT Sentence Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/pankaj90382/END-1.0/blob/main/S14/BERT%20Fine-Tuning%20Sentence%20Classification%20v4.ipynb)



**Training Logs**

```text

======== Epoch 1 / 4 ========
Training...
  Batch    40  of    241.    Elapsed: 0:00:26.
  Batch    80  of    241.    Elapsed: 0:00:53.
  Batch   120  of    241.    Elapsed: 0:01:19.
  Batch   160  of    241.    Elapsed: 0:01:46.
  Batch   200  of    241.    Elapsed: 0:02:13.
  Batch   240  of    241.    Elapsed: 0:02:39.

  Average training loss: 0.48
  Training epcoh took: 0:02:40

Running Validation...
  Accuracy: 0.83
  Validation Loss: 0.42
  Validation took: 0:00:06

======== Epoch 2 / 4 ========
Training...
  Batch    40  of    241.    Elapsed: 0:00:27.
  Batch    80  of    241.    Elapsed: 0:00:53.
  Batch   120  of    241.    Elapsed: 0:01:20.
  Batch   160  of    241.    Elapsed: 0:01:47.
  Batch   200  of    241.    Elapsed: 0:02:13.
  Batch   240  of    241.    Elapsed: 0:02:40.

  Average training loss: 0.31
  Training epcoh took: 0:02:40

Running Validation...
  Accuracy: 0.84
  Validation Loss: 0.47
  Validation took: 0:00:06

======== Epoch 3 / 4 ========
Training...
  Batch    40  of    241.    Elapsed: 0:00:27.
  Batch    80  of    241.    Elapsed: 0:00:53.
  Batch   120  of    241.    Elapsed: 0:01:20.
  Batch   160  of    241.    Elapsed: 0:01:47.
  Batch   200  of    241.    Elapsed: 0:02:13.
  Batch   240  of    241.    Elapsed: 0:02:40.

  Average training loss: 0.19
  Training epcoh took: 0:02:40

Running Validation...
  Accuracy: 0.85
  Validation Loss: 0.50
  Validation took: 0:00:06

======== Epoch 4 / 4 ========
Training...
  Batch    40  of    241.    Elapsed: 0:00:27.
  Batch    80  of    241.    Elapsed: 0:00:53.
  Batch   120  of    241.    Elapsed: 0:01:20.
  Batch   160  of    241.    Elapsed: 0:01:47.
  Batch   200  of    241.    Elapsed: 0:02:13.
  Batch   240  of    241.    Elapsed: 0:02:40.

  Average training loss: 0.14
  Training epcoh took: 0:02:40

Running Validation...
  Accuracy: 0.85
  Validation Loss: 0.56
  Validation took: 0:00:06

Training complete!
Total training took 0:11:04 (h:mm:ss)

```

**Sample Outputs**



```text

sentence  > i am counting on bill to get there on time .
predicted < acceptable
true cls  = acceptable

sentence  > the bird sings .
predicted < acceptable
true cls  = acceptable

sentence  > john ate dinner but i don ' t know who .
predicted < acceptable
true cls  = unacceptable

sentence  > i wonder to whom they dedicated the building .
predicted < acceptable
true cls  = acceptable

sentence  > the bucket was kicked by pat .
predicted < acceptable
true cls  = acceptable

sentence  > susan told a story to her .
predicted < acceptable
true cls  = acceptable

sentence  > i know the person whose mother died .
predicted < acceptable
true cls  = acceptable

sentence  > we gave ourselves to the cause .
predicted < acceptable
true cls  = acceptable

sentence  > chris walks , pat eats bro ##cco ##li , and sandy plays squash .
predicted < acceptable
true cls  = acceptable

sentence  > chocolate eggs were hidden from each other by the children .
predicted < acceptable
true cls  = unacceptable

```

**Training Validation Loss Curve**

![bert classification model training loss plot](./BERT%20Sentence%20Classification%20Loss.png?raw=true)


**Matthews Corr. Coef**

The score will be based on the entire test set, but let's take a look at the scores on the individual batches to get a sense of the variability in the metric between batches. Each batch has 32 sentences in it, except the last batch which has only (516 % 32) = 4 test sentences in it.

![bert MC PLot](./Sentence%20Classification%20MC%20Score.png?raw=true)

---
 <a id="task3"></a>
 ### BART Paraphrasing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/pankaj90382/END-1.0/blob/main/S14/BART_For_Paraphrasing.ipynb)

#### BART

**B**idirectional and **A**uto-**R**egressive **T**ransformers

BART is a denoising autoencoder for pretraining sequence-to-sequence models. BART is trained by 

- corrupting text with an arbitrary noising function, and 
- learning a model to reconstruct the original text.

BART uses a standard Transformer architecture (Encoder-Decoder) and is a combination of BERT, which is only encoder-model and GPT, which is a decoder-only model.

A significant advantage of this setup is the unlimited flexibility of choosing the corruption scheme; including changing the length of the original input. Or, in fancier terms, the text can be corrupted with an arbitrary noising function.

The corruption schemes used in the paper are summarized below:

- Token masking - a random subset of the input is replaced with [MASK] token, like in BERT
- Token deletion - Random tokens are deleted from the input. The model must decide which positions are missing (as the tokens are simply deleted and not replaced with anything else)
- Text infilling - A number of text spans (length can vary) are each replaced with a single [MASK] token. 
- Sentence Permutation - the input is split based on periods (.) and the sentences are shuffled. 
- Document Rotation - a token is chosen at random, and the sequence is rotated so that it starts with the chosen token. 

#### Pretraining: Fill In the Span

BART is trained on tasks where spans of text are replaced by masked tokens, and the model must learn to reconstruct the original document from this altered span of text.

BART improves on BERT by replacing the BERT's fill-in-the-blank cloze task with a more complicated mix of pretraining tasks.

![text infilling](https://miro.medium.com/max/700/1*jWecxbzBsJEbNgiy_LOIvw.png?raw=true)

In the above example the origin text is ` A B C D E` and the span `C, D` is masked before sending it to the encoder, also an extra mask is placed between `A` and `B` and one mask is removed between `B` and `E`, now the corrupted document is `A _ B _ E`. The encoder takes this as input, encodes it and throws it to the decoder.

The decoder must now use this encoding to reconstruct the original document. `A B C D E`

**Training Logs**

```text
INFO:filelock:Lock 140029390796432 acquired on /root/.cache/huggingface/transformers/3f12fb71b844fcb7d591fdd4e55027da90d7b5dd6aa5430ad00ec6d76585f26c.58d5dda9f4e9f44e980adb867b66d9e0cbe3e0c05360cefe3cd86f5db4fff042.lock
Downloading: 100%
1.60k/1.60k [00:00<00:00, 40.2kB/s]
INFO:filelock:Lock 140029390796432 released on /root/.cache/huggingface/transformers/3f12fb71b844fcb7d591fdd4e55027da90d7b5dd6aa5430ad00ec6d76585f26c.58d5dda9f4e9f44e980adb867b66d9e0cbe3e0c05360cefe3cd86f5db4fff042.lock
INFO:filelock:Lock 140029278969424 acquired on /root/.cache/huggingface/transformers/d065edfe6954baf0b989a2063b26eb07e8c4d0b19354b5c74af9a51f5518df6e.6ca4df1a6ec59aa763989ceec10dff41dde19f0f0824b9f5d3fcd35a8abffdb2.lock
Downloading: 100%
1.02G/1.02G [00:22<00:00, 40.5MB/s]
INFO:filelock:Lock 140029278969424 released on /root/.cache/huggingface/transformers/d065edfe6954baf0b989a2063b26eb07e8c4d0b19354b5c74af9a51f5518df6e.6ca4df1a6ec59aa763989ceec10dff41dde19f0f0824b9f5d3fcd35a8abffdb2.lock
INFO:filelock:Lock 140026355392464 acquired on /root/.cache/huggingface/transformers/0d6fc8b2ef1860c1f8f0baff4b021e3426cc7d11b153f98e563b799603ee2f25.647b4548b6d9ea817e82e7a9231a320231a1c9ea24053cc9e758f3fe68216f05.lock
Downloading: 100%
899k/899k [00:00<00:00, 1.66MB/s]
INFO:filelock:Lock 140026355392464 released on /root/.cache/huggingface/transformers/0d6fc8b2ef1860c1f8f0baff4b021e3426cc7d11b153f98e563b799603ee2f25.647b4548b6d9ea817e82e7a9231a320231a1c9ea24053cc9e758f3fe68216f05.lock
INFO:filelock:Lock 140026355391888 acquired on /root/.cache/huggingface/transformers/6e75e35f0bdd15870c98387e13b93a8e100237eb33ad99c36277a0562bd6d850.5d12962c5ee615a4c803841266e9c3be9a691a924f72d395d3a6c6c81157788b.lock
Downloading: 100%
456k/456k [00:00<00:00, 1.52MB/s]
INFO:filelock:Lock 140026355391888 released on /root/.cache/huggingface/transformers/6e75e35f0bdd15870c98387e13b93a8e100237eb33ad99c36277a0562bd6d850.5d12962c5ee615a4c803841266e9c3be9a691a924f72d395d3a6c6c81157788b.lock
INFO:filelock:Lock 140026355390736 acquired on /root/.cache/huggingface/transformers/d94f53c8851dcda40774f97280e634b94b721a58e71bcc152b5f51d0d49a046a.fc9576039592f026ad76a1c231b89aee8668488c671dfbe6616bab2ed298d730.lock
Downloading: 100%
1.36M/1.36M [00:00<00:00, 4.87MB/s]
INFO:filelock:Lock 140026355390736 released on /root/.cache/huggingface/transformers/d94f53c8851dcda40774f97280e634b94b721a58e71bcc152b5f51d0d49a046a.fc9576039592f026ad76a1c231b89aee8668488c671dfbe6616bab2ed298d730.lock
INFO:filelock:Lock 140026356565712 acquired on /root/.cache/huggingface/transformers/1abf196c889c24daca2909359ca2090e5fcbfa21a9ea36d763f70adbafb500d7.67d01b18f2079bd75eac0b2f2e7235768c7f26bd728e7a855a1c5acae01a91a8.lock
Downloading: 100%
26.0/26.0 [00:00<00:00, 669B/s]
INFO:filelock:Lock 140026356565712 released on /root/.cache/huggingface/transformers/1abf196c889c24daca2909359ca2090e5fcbfa21a9ea36d763f70adbafb500d7.67d01b18f2079bd75eac0b2f2e7235768c7f26bd728e7a855a1c5acae01a91a8.lock
INFO:simpletransformers.seq2seq.seq2seq_utils: Creating features from dataset file at cache_dir/
100%
21829/21829 [00:08<00:00, 2639.14it/s]
INFO:simpletransformers.seq2seq.seq2seq_model: Training started
Epoch 1 of 1: 100%
1/1 [52:28<00:00, 3148.78s/it]
wandb: You can find your API key in your browser here: https://wandb.ai/authorize
wandb: Paste an API key from your profile and hit enter: ··········
wandb: W&B syncing is set to `offline` in this directory.  Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
Epochs 0/1. Running Loss: 0.6717: 100%
2729/2729 [46:27<00:00, 1.18it/s]
INFO:simpletransformers.seq2seq.seq2seq_utils: Creating features from dataset file at cache_dir/
100%
3539/3539 [00:07<00:00, 301.58it/s]
INFO:simpletransformers.seq2seq.seq2seq_model:{'eval_loss': 0.4942293555081428}
INFO:simpletransformers.seq2seq.seq2seq_model:Saving model into outputs/best_model
INFO:simpletransformers.seq2seq.seq2seq_model:Saving model into outputs/checkpoint-2729-epoch-1
INFO:simpletransformers.seq2seq.seq2seq_utils: Creating features from dataset file at cache_dir/
100%
3539/3539 [00:11<00:00, 205.02it/s]
INFO:simpletransformers.seq2seq.seq2seq_model:{'eval_loss': 0.4932126947888383}
INFO:simpletransformers.seq2seq.seq2seq_model:Saving model into outputs/best_model
INFO:simpletransformers.seq2seq.seq2seq_model:Saving model into outputs/
INFO:simpletransformers.seq2seq.seq2seq_model: Training of facebook/bart-large model complete. Saved to outputs/.
Generating outputs: 100%
222/222 [17:36<00:00, 3.62s/it]
/usr/local/lib/python3.7/dist-packages/torch/_tensor.py:575: UserWarning: floor_divide is deprecated, and will be removed in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values.
To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor'). (Triggered internally at  /pytorch/aten/src/ATen/native/BinaryOps.cpp:467.)
  return torch.floor_divide(self, other)
```

**Sample Outputs**

```text
Text  > They were there to enjoy us and they were there to pray for us .
Pred  < Paraphrase : They were there to enjoy us and they were there for us to pray.
Truth = They were there for us to enjoy and they were there for us to pray .

Text  > After the end of the war in June 1902 , Higgins left Southampton in the `` SSBavarian '' in August , returning to Cape Town the following month .
Pred  < Paraphrase : After the end of the war in June 1902, Higgins left Southampton in August in the `` SSBavarian '', returning to Cape Town the following month.
Truth = In August , after the end of the war in June 1902 , Higgins Southampton left the `` SSBavarian '' and returned to Cape Town the following month .

Text  > From the merger of the Four Rivers Council and the Audubon Council , the Shawnee Trails Council was born .
Pred  < Paraphrase : From the merger of the Four Rivers Council and the Audubon Council, the Shawnee Trails Council was born.
Truth = Shawnee Trails Council was formed from the merger of the Four Rivers Council and the Audubon Council .

Text  > The group toured extensively and became famous in Israel , and even played in New York City in 2007 .
Pred  < Paraphrase : The group toured extensively and became famous in Israel, and even played in New York City in 2007.
Truth = The group toured extensively and was famous in Israel and even played in New York City in 2007 .

Text  > Kathy and her husband Pete Beale ( Peter Dean ) are stable financially .
Pred  < Paraphrase : Kathy and her husband Peter Dean ( Pete Beale ) are financially stable.
Truth = Kathy and her husband Peter Dean ( Pete Beale ) are financially stable .

Text  > Timora diarhoda is a species of moth of the Noctuidae family . It is found in Africa , including South Africa .
Pred  < Paraphrase : Timora diarhoda is a species of moth of the Noctuidae family. It is found in Africa, including South Africa.
Truth = Diarhoda is a kind of moth of the Noctuidae family . It is found in South Africa including Africa .

Text  > Joe R. Campa Jr. is a former sailor of the United States Navy , who served as the eleventh Master Chief Petty Officer of the U.S. Navy .
Pred  < Joe R. Campa Jr. is a former sailor of the United States Navy, who served as the eleventh Master Chief Petty Officer of the Navy.
Truth = Joe R. Campa Jr. is a former U.S. Navy Matrose who served as the 11th Master Chief Petty Officer of the United States Navy .

Text  > The exception was between late 2005 and 2009 when he played in Sweden with Carlstad United BK , Serbia with FK Borac Čačak and Russian FC Terek Grozny .
Pred  < The exception was between late 2005 and 2009 when he played with Carlstad United BK in Sweden, Serbia with FK Borac Čačak and Russian FC Terek Grozny.
Truth = The exception was between late 2005 and 2009 , when he played in Sweden with Carlstad United BK , Serbia with FK Borac Čačak and the Russian FC Terek Grozny .

Text  > The family moved to Camp Hill in 1972 , where he attended Trinity High School in Harrisburg , Pennsylvania .
Pred  < Paraphrase : The family moved to Camp Hill in 1972, where he attended Trinity High School in Harrisburg, Pennsylvania.
Truth = In 1972 , the family moved to Camp Hill , where he visited the Trinity High School in Harrisburg , Pennsylvania .

Text  > A recording of folk songs done for the Columbia society in 1942 was largely arranged by Pjetër Dungu .
Pred  < Paraphrase : A recording of folk songs done in 1942 for the Columbia Society was largely arranged by Pjetër Dungu.
Truth = A recording of folk songs made for the Columbia society in 1942 was largely arranged by Pjetër Dungu .

Text  > Components of elastic potential systems store mechanical energy if they are deformed when forces are applied to the system .
Pred  < Paraphrase : Components of elastic potential systems store mechanical energy if they are deformed when forces are applied to the system.
Truth = Components of elastic potential systems store mechanical energy if they are deformed to the system when applied to forces .

Text  > Earl St Vincent was a British ship that was captured in 1803 and became a French trade man .
Pred  < Earl St Vincent was a British ship captured in 1803 and became a French trade man.
Truth = Earl St Vincent was a British ship that was captured and became a French merchantman in 1803 .

Text  > Steam can also be used , and does not need to be pumped .
Pred  < Paraphrase : Steam can also be used and does not need to be pumped.
Truth = Also steam can be used and need not be pumped .

Text  > In mathematical astronomy , his fame is due to the introduction of the astronomical globe , and his early contributions to understanding the movement of the planets .
Pred  < Paraphrase : In mathematical astronomy, his fame is due to the introduction of the astronomical globe and his early contributions to understanding the movement of the planets.
Truth = His fame is due in mathematical astronomy to the introduction of the astronomical globe and to his early contributions to the understanding of the movement of the planets .

Text  > Winarsky is a member of the IEEE , Phi Beta Kappa , the ACM and Sigma Xi .
Pred  < Paraphrase : Winarsky is a member of the ACM, Phi Beta Kappa, the IEEE and Sigma Xi.
Truth = Winarsky is a member of ACM , the IEEE , the Phi Beta Kappa and the Sigma Xi .

Text  > Mandarin - Chinese has specific terms and racial euphemisms for different races and ethnicities , and some discriminatory attacks against representatives of certain governments and backgrounds .
Pred  < Paraphrase : Mandarin Chinese has specific terms and racial euphemisms for different races and ethnicities, and some discriminatory attacks against representatives of certain governments and backgrounds.
Truth = Mandarin Chinese has specific terms and racial euphemisms for different races and ethnicities , and some discriminatory slurs against representatives from certain governments and backgrounds .

Text  > The solar approach to this requirement is the use of solar panels in a conventional-powered aircraft .
Pred  < Paraphrase : The solar approach to this requirement is the use of solar panels in a conventional aircraft.
Truth = The solar approach to this requirement is the use of solar panels in a conventionally powered aircraft .

Text  > The police also questioned singer Rimi Tomy and the actor Kavya Madhavan , both close friends of Siddique and his wife Dileep , as part of the ongoing investigation .
Pred  < Paraphrase : The police also questioned singer Rimi Tomy and actor Kavya Madhavan, both close friends of Siddique and his wife Dileep, as part of the ongoing investigation.
Truth = The police also questioned singer Rimi Tomy and actor Kavya Madhavan , both close friends of Siddique and his wife Dileep , as part of the ongoing investigation .

Text  > Its music critics included Julius Korngold ( 1864 -- 1904 ) and Eduard Hanslick ( 1904 -- 1934 ) .
Pred  < Paraphrase : Its music critics included Julius Korngold ( 1864 -- 1904 ) and Eduard Hanslick ( 1904 -- 1934 ).
Truth = Its music critics included Julius Korngold ( 1864 -- 1904 ) and Eduard Hanslick ( 1904 -- 1934 ) .

Text  > Holly was musically influenced by Elton John .
Pred  < Paraphrase : Holly was musically influenced by Elton John.
Truth = Holly Holly was influenced musically by Elton John .

Text  > It was that Easipower said ,
Pred  < Paraphrase : It was Easipower that said,
Truth = It was Easipower that said :

Text  > The Tabaci River is a tributary of the Leurda River in Romania .
Pred  < The Tabaci River is a tributary of the Leurda River in Romania.
Truth = The river Tabaci is a tributary of the River Leurda in Romania .

Text  > Aamir Khan agreed to act immediately after reading Mehra 's screenplay in `` Rang De Basanti '' .
Pred  < Paraphrase : Aamir Khan agreed to act immediately after reading Mehra's screenplay in `` Rang De Basanti ''.
Truth = Aamir Khan agreed to act in `` Rang De Basanti '' immediately after reading Mehra 's script .

Text  > Loyang 's team defeated Xinmin Secondary School in the opening round and Raffles Institution in the quarter-finals , before losing to Hwa Chong Institution in the semi-finals .
Pred  < Paraphrase : Loyang's team defeated Xinmin Secondary School in the opening round and Raffles Institution in the quarter-finals before losing in the semi-finals to Hwa Chong Institution.
Truth = The Loyang team defeated Xinmin Secondary School in the opening round and Raffles Institution in the quarterfinals before losing to the Hwa Chong Institution in the semi-finals .

Text  > Retzius was born in Stockholm , son of the anatomist Anders Jahan Retzius ( and grandson of the naturalist and chemist Anders Retzius ) .
Pred  < Paraphrase : Retzius was born in Stockholm, son of anatomist Anders Jahan Retziu ( and grandson of the naturalist and chemist Anders Retzlius ).
Truth = Retzius was born in Stockholm , the son of anatomist Anders Jahan Retzius ( and grandson of the naturalist and chemist Anders Retzius ) .

Text  > The `` Fallbeil '' was used for the last time in West Germany in 1949 , in East Germany in 1966 .
Pred  < Paraphrase : The `` Fallbeil '' was used for the last time in 1949 in West Germany, in 1966 in East Germany.
Truth = The `` Fall Beil '' was used for the last time in West Germany in 1949 , in 1966 in East Germany .

Text  > In 2009 he moved back to Philadelphia and lives in New York City today .
Pred  < Paraphrase : In 2009 he moved back to Philadelphia and lives in New York City today.
Truth = He moved back to Philadelphia in 2009 and now lives in New York City .

Text  > Stipsits was born in Korneuburg , Germany and spent his childhood in Stammersdorf , Vienna .
Pred  < Stipsits was born in Korneuburg, Germany, and spent his childhood in Stammersdorf, Vienna.
Truth = Stipsits was born in Korneuburg , and spent his childhood in Stammersdorf , Vienna .

Text  > Daudkhali is a village in Barisal Division in the Pirojpur district in southwestern Bangladesh .
Pred  < Daudkhali is a village in the Barisal Division in the Pirojpur district of southwestern Bangladesh.
Truth = Daudkhali is a village in Pirojpur District in the Barisal Division of southwestern Bangladesh .

Text  > The temple serves as the cultural and religious center for Korean Hindus and immigrants from South Asian countries .
Pred  < Paraphrase : The temple serves as the cultural and religious center for Korean Hindus and immigrants from South Asian countries.
Truth = The temple serves as a cultural and religious centre for Korean Hindus and immigrants from the South Asian countries .

Text  > When a surface has a constant zero developable curvature , then it is a Euclidean surface and the geometry of the surface is Gaussian geometry .
Pred  < Paraphrase : When a surface has a constant zero developable curvature, then it is a Euclidean surface and the geometry of the surface is Gaussian geometry.
Truth = If a surface has a constant developable curvature of zero , then it is an euclidean surface and the geometry of the surface is a Gaussian geometry .

Text  > The following is a list of fouls established by the states that regulate MMA , as outlined by the Nevada State Athletic Commission .
Pred  < The following is a list of fouls established by the states that regulate MMA as outlined by the Nevada State Athletic Commission.
Truth = The following is a list of fouls defined by the states that regulate MMA as outlined by the Nevada State Athletic Commission .

Text  > The Keita dynasty ruled pre-imperial and imperial Mali from the 12th century into the early 17th century .
Pred  < Paraphrase : The Keita dynasty ruled pre-imperial and imperial Mali from the 12th century to the early 17th century.
Truth = The Keita dynasty ruled Mali from the 12th to the early 17th century , pre-imperial and imperial .

Text  > The compound was patented by Dr. Patrick Page and his team and was invented by Genkyotex in 2007 .
Pred  < Paraphrase : The compound was patented by Dr. Patrick Page and his team and was invented by Genkyotex in 2007.
Truth = The compound was patented by Dr. Patrick Page and his team , and was invented in 2007 by Genkyotex .

Text  > When Phil Spector heard `` Stubborn Kind of Fellow '' for the first time , he was so excited that he lost control of his car while driving down the Sunset Boulevard with Jack Nitzsche .
Pred  < Paraphrase : When Phil Spector heard `` Stubborn Kind of Fellow '' for the first time, he was so excited that he lost control of his car while driving with Jack Nitzsche on Sunset Boulevard.
Truth = When Phil Spector first heard `` Stubborn Kind of Fellow '' he was so excited he lost control of his car while driving down Sunset Boulevard with Jack Nitzsche .

Text  > This genus is presently in the family of lizards known as Iguanidae , subfamily Polychrotinae , and is no longer classified in the now invalid family , Polychrotidae .
Pred  < Paraphrase : This genus is currently in the family of lizards known as Iguanidae, subfamily Polychrotinae, and is no longer classified in the now invalid family PolyChrotidae.
Truth = This genus is currently classified in the family of lizards , known as Iguanidae , subfamily Polychrotinae , and is no longer classified in the now invalid family , polychrotidae .

Text  > Iyer next appeared in the Kannada film `` Darshan '' with actor Jaggu Dada .
Pred  < Paraphrase : Iyer next appeared in the Kannada film `` Darshan '' with the actor Jaggu Dada.
Truth = Next appeared Iyer in Kannada - Film `` Darshan '' with actor Jaggu Dada .

Text  > Born in South Korea , he lived in Brazil for 9 years since 2002 and played football there for 5 years . He started his professional career in 2007 .
Pred  < Paraphrase : Born in South Korea, he lived in Brazil for 9 years since 2002 and played football there for 5 years. He started his professional career in 2007.
Truth = Born in South Korea , he lived since 2002 for 9 years in Brazil , played there 5 years of football and started his career in 2007 .

Text  > On May 8 , 2015 , Michel Soro lost to Tapia .
Pred  < Paraphrase : Michel Soro lost to Tapia on 8 May 2015.
Truth = On 8 May 2015 , Michel Soro lost Tapia .

Text  > The music was written by Shyam and lyrics was composed by Sreekumaran Thampi and Sathyan Anthikkad .
Pred  < Paraphrase : The music was written by Shyam and lyrics was composed by Sreekumaran Thampi and Sathyan Anthikkad.
Truth = The music was written by Shyam and the lyrics by Sreekumaran Thampi and Sathyan Anthikkad composed .

Text  > In 2012 , Ned Evett released `` Treehouse '' , his sixth solo record , produced in Nashville Tennessee by musician Adrian Belew .
Pred  < In 2012, Ned Evett released `` Treehouse '', his sixth solo record, produced by the musician Adrian Belew in Nashville Tennessee.
Truth = In 2012 , Ned Evett `` Treehouse '' released his sixth solo record , produced in Nashville Tennessee by musician Adrian Belew .

Text  > North Northamptonshire was a county constituency in Northamptonshire , represented in the House of Commons of the Parliament of the United Kingdom .
Pred  < North Northamptonshire was a county constituency in Northamptonhire, represented in the House of Commons of the Parliament of the United Kingdom.
Truth = Northamptonshire was a county constituency in Northamptonshire , represented in the House of Commons of the United Kingdom Parliament .

Text  > In February 2014 , Network Ten announced that Hugh Riminton would replace Danielle Isdale as presenter , and Victoria Murphy would become the sports presenter .
Pred  < In February 2014, Network Ten announced that Hugh Riminton would replace Danielle Isdale as presenter, and Victoria Murphy would become the sports presenter.
Truth = In February 2014 , Network Ten announced that Hugh Riminton would replace Danielle Isdale as presenter and Victoria Murphy as sports presenter .

Text  > The crash of Aerosucre Flight 4544 was the second aviation incident involving Aerosucre , the first being the crash of another Boeing 727 on November 18 , 2006 .
Pred  < The crash of Aerosucre Flight 4544 was the second aviation incident involving Aerosüre, the first being the crash of another Boeing 727 on November 18, 2006.
Truth = The crash of Aerosucre Flight 4544 was the second aviation incident with Aerosucre , the first being the crash of another Boeing 727 on November 18 , 2006 .

Text  > Among them are Marie Teresa Rios , a composer of Boleros , Julita Ross , author , and Sylvia Rexach , a singer .
Pred  < Among them are Marie Teresa Rios, a composer of Boleros, Julita Ross, author and Sylvia Rexach, a singer.
Truth = Among them are Marie Teresa Rios , a composer of boleros , Julita Ross , an author , and Sylvia Rexach , a singer .

Text  > Kabir asks Sarika to reveal his plan to end Ranvir 's game .
Pred  < Paraphrase : Kabir asks Sarika to reveal his plan to end Ranvir's game.
Truth = Kabir asks Sarika to reveal his plan to end Ranvir 's game to him .

Text  > `` Trust in Me '' is a song written by Ned Wever , Milton Ager , and Jean Schwartz .
Pred  < `` Trust in Me '' is a song written by Ned Wever, Milton Ager and Jean Schwartz.
Truth = `` Trust in Me '' is a song by Jean Schwartz , Milton Ager and Ned Wever written .

Text  > Jason Thornton is the orchestra 's Artistic Director , Peter Donohoe the Principal Guest Conductor and Gavin Carr the Associate Conductor .
Pred  < Paraphrase : Jason Thornton is the orchestra's Artistic Director, Peter Donohoe the Principal Guest Conductor and Gavin Carr the Associate Conductor.
Truth = Jason Thornton is the artistic director of the orchestra , Peter Donohoe , the principal guest conductor and Gavin Carr of the Associate Conductor .

Text  > Bob and Ted were brothers . Ted is John 's son .
Pred  < Paraphrase : Bob and Ted were brothers, Ted is John's son.
Truth = Bob and Ted were brothers , and Ted is John 's son .

Text  > It was released on 22 December 2011 and was announced in February 2012 .
Pred  < Paraphrase : It was released on December 22, 2011 and announced in February 2012.
Truth = It was published on 22 December 2011 and was announced in February 2012 .

Text  > From the west end of the bridge , Pennsylvania Route 268 leads south to Parker and north to Emlenton .
Pred  < Paraphrase : From the west end of the bridge, Pennsylvania Route 268 leads south to Parker and north to Emlenton.
Truth = The Pennsylvania Route 268 leads from the west end of the bridge south to Parker and to the north to Emlenton .

Text  > The Loyang team defeated the Raffles Institution in the opening round and the Xinmin Secondary School in the quarter-finals before losing to the Hwa Chong Institution in the semi-finals .
Pred  < Paraphrase : The Loyang team defeated the Raffles Institution in the opening round and the Xinmin Secondary School in the quarterfinals before losing in the semi-finals to the Hwa Chong Institution.
Truth = Loyang 's team defeated Raffles Institution in the opening round and Xinmin Secondary School in the quarter-finals , before losing to Hwa Chong Institution in the semi-finals .

Text  > Alston was born on December 21 , 1965 in Oxon Hill , Maryland . He attended Oxon Hill High School in New Haven , Connecticut .
Pred  < Paraphrase : Alston was born on 21 December 1965 in Oxon Hill, Maryland. He attended Oxon Hills High School in New Haven, Connecticut.
Truth = He was born on December 21 , 1965 in Oxon Hill , Maryland , and attended High School in New Haven , Connecticut .

Text  > Total US casualties were 28 killed , while Viet Cong losses were 345 killed and a further 192 estimated killed .
Pred  < Paraphrase : Total US casualties were 28 killed, while Viet Cong losses were 345 killed and a further 192 killed.
Truth = In total , 28 US victims were killed , while Viet Cong losses were killed 345 and a further 192 estimated killed .

Text  > In CA , the title of a chartered accountant ( Sri Lanka Sri Lanka ) can only be used by members of the Institute of Sri Lankan Accountants .
Pred  < Paraphrase : In CA, the title of Chartered Accountant ( Sri Lanka Sri Lanka ) can only be used by members of the Institute of Sri Lankan Accountants.
Truth = In CA , the title of Chartered Accountant ( Sri Lanka Sri Lanka ) can be used by only members of the Institute of Chartered Accountants of Sri Lanka .

Text  > Simyo belongs to the Dutch telecommunications group KPN , after acquisition of the remainder of E-Plus on March 14 .
Pred  < Paraphrase : Simyo belongs to the Dutch telecommunications group KPN after the acquisition of the remainder of E-Plus on 14 March.
Truth = Following the acquisition of the remainder of E-Plus on 14 March , Simyo belongs to the Dutch telecommunications group KPN .

Text  > Worcester is a town and county city of Worcestershire in England .
Pred  < Paraphrase : Worcester is a town and county town of Worcestershire in England.
Truth = Worcester is a city and county town of Worcestershire in England .

Text  > Besides Kuykendall , Robert White and Joshua Soule Zimmerman served as Chancery Commissioner for Hampshire County .
Pred  < Paraphrase : Besides Kuykendall, Robert White and Joshua Soule Zimmerman served as Chancery Commissioner for Hampshire County.
Truth = Robert White and Joshua Soule Zimmerman served alongside Kuykendall as a Chancery Commissioner for Hampshire County .

Text  > In Sri Lanka , the title of Chartered Accountant ( CA Sri Lanka ) can be used by only members of the Institute of Chartered Accountants of Sri Lanka .
Pred  < Paraphrase : In Sri Lanka, the title of Chartered Accountant ( CA Sri Lanka ) can be used only by members of the Institute of Chartering Accountants of Sri Lanka.
Truth = In Sri Lanka , the title of an accountant ( CA Sri Lanka ) can only be used by members of the Institute of Accountants in Sri Lanka .

Text  > July is the coldest month on average , the hottest in January .
Pred  < Paraphrase : July is the coldest month on average, the hottest in January.
Truth = July is on average the coldest month and January the hottest .

Text  > He was born in Carter County , Tennessee and was later moved to Arkansas .
Pred  < Paraphrase : He was born in Carter County, Tennessee, and later moved to Arkansas.
Truth = He was born in Carter County , Tennessee and later moved to Arkansas .

Text  > These views were often expressed during the emergence of Protestant , Puritan and Evangelical movements .
Pred  < These views were often expressed during the emergence of the Protestant, Puritan and Evangelical movements.
Truth = These views were often expressed during the emergence of evangelical , puritanical , and protestant movements .

Text  > He died in Madrid on 6 January 1976 , and was buried in the church of the in Alicante .
Pred  < Paraphrase : He died on 6 January 1976 in Madrid and was buried in the church of Alicante.
Truth = He died on 6 January 1976 in Madrid and was buried in the church of Alicante .

Text  > In 1876 , he moved to San Diego , California , and in 1887 to Dallas , Texas .
Pred  < Paraphrase : In 1876 he moved to San Diego, California, and in 1887 to Dallas, Texas.
Truth = He moved to San Diego , California in 1876 , and to Dallas , Texas in 1887 .

Text  > The UEFA Cup 1973 -- 74 was won by Feyenoord Rotterdam on Tottenham Hotspur 4 : 2 .
Pred  < Paraphrase : The UEFA Cup 1973 -- 74 was won by Feyenoord Rotterdam on Tottenham Hotspur 4 : 2.
Truth = The 1973 -- 74 UEFA Cup was won by Feyenoord Rotterdam over Tottenham Hotspur 4 -- 2 on aggregate .

Text  > He married Marie Magdalene Schweigaard , daughter of Tellef Dahll Schweigaard , niece of leading politician Anton Martin Schweigaard and aunt of later Prime Minister Christian Homann Schweigaard .
Pred  < Paraphrase : He married Marie Magdalene Schweigaard, daughter of Tellef Dahll Schweigaad, niece of leading politician Anton Martin Schweigaards and aunt of later Prime Minister Christian Homann SchweigaARD.
Truth = He married Marie Magdalene Schweigaard , daughter of Tellef Dahll Schweigaard , niece of the senior politician Anton Martin Schweigaard and Tante of later prime minister Christian Homann Schweigaard .

Text  > Rõuge Valgjärv is a lake in the southeastern county of Voru in Estonia , close to the border with Latvia .
Pred  < Rõuge Valgjärv is a lake in the southeastern Voru County in Estonia, close to the border with Latvia.
Truth = Rõuge Valgjärv is a lake in Estonia 's southeastern county of Voru , close to the border with Latvia .

Text  > The Central Baptist Association is an association of churches located from South Carolina to Indiana , with most of the churches being in eastern Tennessee and southwestern Virginia .
Pred  < Paraphrase : The Central Baptist Association is an association of churches from South Carolina to Indiana, with most of the churches located in eastern Tennessee and southwestern Virginia.
Truth = The Central Baptist Association is an association of churches from South Carolina to Indiana , with the most churches in eastern Tennessee and south-western Virginia .

Text  > It was part of the Hanover Township , then Chatham Township , before being recorded in 1899 as Florham Park .
Pred  < Paraphrase : It was part of Hanover Township, then Chatham Township, before being recorded as Florham Park in 1899.
Truth = It was part of Hanover Township , then Chatham Township before being incorporated as Florham Park in 1899 .

Text  > They also released the second track on the album , `` Vices '' , as the 5th single from the album on June 13 .
Pred  < They also released the second track on the album `` Vices '' as the 5th single from the album on 13 June.
Truth = They also released the second track on the album , `` Vices '' , on 13th June as the 5th single from the album .

Text  > The loyalists had camped on the west side of the Catawba River , while the army of General Charles Cornwalli camped on the east side .
Pred  < Paraphrase : The loyalists had camped on the west side of the Catawba River, while the army of General Charles Cornwalli was on the east side.
Truth = The loyalists were camped on the west side of Catawba River , while the army of General Charles Cornwallis camped on the eastern side .

Text  > The Sydney Water Board had taken over the water supply for Sydney from the City Council in 1888 .
Pred  < Paraphrase : The Sydney Water Board had taken over the water supply for Sydney from the City Council in 1888.
Truth = In 1888 , the Sydney Water Board took over the water supply for Sydney from the city council .

Text  > Mohammad Shafiq ( variants : Mohammed , Muhammad , Shafik , Shafeek , Shafeeq , Shafique , Shafic , Chafic ) can refer to
Pred  < Paraphrase : Mohammad Shafiq ( variants : Shafik, Shafeek, Shafeeq, Muhammad, Mohammed, Muhammad ; Shafic, Chafic ) can refer to
Truth = Mohammad Shafiq ( variants : Mohammed , Muhammad , Shafik , Shafeek , Shafeeq , Shafique , Shafic , Chafic ) may refer to

Text  > `` T '' is similar to a matrix whose canonical entries are on the superdiagonal , by the Jordan only non-zero form .
Pred  < Paraphrase : `` T '' is similar to a matrix whose canonical entries are on the superdiagonal, by the Jordan only non-zero form.
Truth = `` T '' is similar to a matrix whose canonical entries are on the superdiagonal form of the Jordan non-zero .

Text  > Fanwood is located in the 22nd Congressional District and is part of the 12th New Jersey Legislative District .
Pred  < Paraphrase : Fanwood is located in the 22nd Congressional District and is part of the 12th New Jersey Legislative District.
Truth = Fanwood is located in the 22nd Congressional District and is part of New Jersey 's 12th state legislative district .

Text  > In 1994 , Rodrigo Leão left the band to start a solo career , being replaced by Carlos Maria Trindade ( keyboard synthesizer ) .
Pred  < In 1994, Rodrigo Leão left the band to start a solo career, being replaced by Carlos Maria Trindade ( keyboard synthesizer ).
Truth = In 1994 , Rodrigo Leão left the band to start a solo career , replaced by Carlos Maria Trindade ( keyboard synthesizer ) .

Text  > Note also that `` not NFL '' implies only that algorithms are inequivalent overall by `` some '' measure of performance .
Pred  < Paraphrase : Note also that `` not NFL '' implies only that algorithms are inequivalent overall by `` some '' measure of performance.
Truth = Note also that `` not NFL '' only implies that algorithms are inequivalent by `` some '' performance dimensions overall .

Text  > Ned Lambton and McEwen divorced in 1995 . She has since married the musician Jools Holland .
Pred  < Paraphrase : Ned Lambton and McEwen divorced in 1995. She has since married the musician Jools Holland.
Truth = Ned Lambton and McEwen divorced in 1995 , and since then she has married the musician Jools Holland .

Text  > In 1977 , Rob Taylor travelled with Barber to Scotland and Norway to climb waterfalls .
Pred  < Paraphrase : In 1977, Rob Taylor travelled with Barber to Scotland and Norway to climb waterfalls.
Truth = In 1977 , Rob Taylor traveled to Scotland and Norway with Barber to climb waterfalls .

Text  > It then crosses over the Washita River arm of Lake Texoma .
Pred  < Paraphrase : It then crosses over the Washita River arm of Lake Texoma.
Truth = It then crosses the Washita River arm of Lake Texoma .

Text  > In 1933 Cattell wrote that , of all the Nordic races , `` the European race was the most evolved in intelligence and stability of temperament '' .
Pred  < Paraphrase : In 1933, Cattell wrote that of all the Nordic races, `` the European race was the most evolved in intelligence and stability of temperament ''.
Truth = In 1933 , Cattell wrote that of all the Nordic races , `` the European race in intelligence and stability of temperament was most developed '' .

Text  > Podkriváň is a village and municipality in the region Banská Bystrica , in the district of Detva in Central Slovakia .
Pred  < Podkriváň is a village and municipality in the Banská Bystrica region in the Detva district of Central Slovakia.
Truth = Podkriváň is a village and municipality in Banská Bystrica Region , in the Detva District of central Slovakia .

Text  > The Russian cavalry withdrew behind the main line and exposed the French to the artillery fire from Russian batteries .
Pred  < Paraphrase : The Russian cavalry withdrew behind the main line and exposed the French to artillery fire from Russian batteries.
Truth = The Russian cavalry withdrew behind the main line , exposing the French to artillery fire from the Russian batteries .

Text  > In 2016 , `` Forbes '' ranked California Maritime Academy as the 95th best university in the nation and 516th in the West .
Pred  < Paraphrase : In 2016, `` Forbes '' ranked California Maritime Academy as the 95th best university in the nation and 516th in the West.
Truth = In 2016 , `` Forbes '' California Maritime Academy ranks as the 95th best university in the nation and 516th in the west .

Text  > Callum O 'brien ( born November 4 , 1982 in Cambridge , New Zealand ) is a professional squash player .
Pred  < Callum O 'brien ( born 4 November 1982 in Cambridge, New Zealand ) is a professional squash player.
Truth = Callum O'brien ( born 4 November 1982 in Cambridge ) is a New Zealand professional squash player .

Text  > Abies lasiocarpa , commonly called the western North American fir or Rocky Mountain fir , is a subalpine fir tree .
Pred  < Abies lasiocarpa, commonly called the western North American fir or Rocky Mountain fir, is a subalpine fir tree.
Truth = Abies lasiocarpa , commonly known as the western North American fir or Rocky Mountain fir , is a subalpine fir tree .

Text  > In malignant hypertension these hyperplastic changes are often accompanied by fibrinoid necrosis of the arterial intima and media .
Pred  < Paraphrase : In malignant hypertension these hyperplastic changes are often accompanied by fibrinoid necrosis of the arterial intima and media.
Truth = In malignant hypertension , these hyperplastic changes are often accompanied by a fibrinoid necrosis of arterial intima and media .

Text  > American Motors provided only technical support in the form of limited aid .
Pred  < Paraphrase : American Motors provided only technical support in the form of limited aid.
Truth = American Motors provided only technical support in the form of limited help .

Text  > It then crosses over the Washita River arm of Lake Texoma .
Pred  < Paraphrase : It then crosses over the Washita River arm of Lake Texoma.
Truth = It then crosses the Washita River Arm of the Texoma Lake .

Text  > The Houston Main Building ( HMB ) earlier the Prudential Building was a skyscraper at the Texas Medical Center in Houston , Texas .
Pred  < Paraphrase : The Houston Main Building ( HMB ) earlier the Prudential Building was a skyscraper at the Texas Medical Center in Houston, Texas.
Truth = The Houston Main Building ( HMB ) formerly the Prudential Building , was a skyscraper in the Texas Medical Center , Houston , Texas .

Text  > The UEFA Cup 1973 -- 74 was won by Tottenham Hotspur over Feyenoord Rotterdam 4 : 2 .
Pred  < Paraphrase : The UEFA Cup 1973 -- 74 was won by Tottenham Hotspur over Feyenoord Rotterdam 4 : 2.
Truth = The 1973 -- 74 UEFA Cup was won by Tottenham Hotspur over Feyenoord Rotterdam 4 -- 2 on aggregate .

Text  > In 1974 Lao PDR established the Stage II fund with the help of the United Nations , the Asian Development Bank , and the World Bank .
Pred  < Paraphrase : In 1974, Lao PDR established the Stage II Fund with the help of the World Bank, the Asian Development Bank and the United Nations.
Truth = In 1974 , with the support of the United Nations , the Asian Development Bank and the World Bank , Lao PDR founded the Stage II Fund .

Text  > The third line of the last verse was changed to `` Kralja Aleksandra , Bože hrani , '' during the reign of Alexander I of Yugoslavia .
Pred  < Paraphrase : The third line of the last verse was changed during the reign of Alexander I of Yugoslavia to `` Kralja Aleksandra, Bože hrani ''.
Truth = The third line of the last verse was modified during the reign of Alexander I of Yugoslavia in `` Kralja Aleksandra , Bože hrani '' .

Text  > Ray Bloch was the announcer , and John Reed King led the orchestra .
Pred  < Paraphrase : Ray Bloch was the announcer and John Reed King led the orchestra.
Truth = The announcer was Ray Ray Bloch , John Reed King led the orchestra .

Text  > Archbishop Seóighe was appointed on 16 May 1485 and consecrated in 1487 . He died on either the 20 or 20 December 1501 .
Pred  < Paraphrase : Archbishop Seóighe was appointed on 16 May 1485 and consecrated in 1487. He died on 20 December or 20 December 1501.
Truth = Archbishop Seóighe was appointed on May 16 , 1485 and consecrated in 1487 , died either on December 20 or 20 , 1501 .

Text  > Another way to regulate the population of deer is to control the birth rate .
Pred  < Paraphrase : Another way to regulate the population of deer is to control the birth rate.
Truth = Another way to regulate deer population is to control the birth rate .

Text  > The Hudeasa River is a tributary of the Bradu River in Romania .
Pred  < The Hudeasa River is a tributary of the River Bradu in Romania.
Truth = The Hudeasa River is the tributary of the Bradu River in Romania .

Text  > The Loyalists had camped on the west side of the Catawba River while General Charles Cornwalli 's army were camped on the east side .
Pred  < Paraphrase : The Loyalists had camped on the west side of the Catawba River, while General Charles Cornwalli's army was on the east side.
Truth = The loyalists were camped on the west side of Catawba River , while the army of General Charles Cornwallis camped on the eastern side .

Text  > His father Patrick Byrne was an MP , TD , Senator and Lord Mayor of Dublin . Another brother Alfie Byrne was also a TD .
Pred  < Paraphrase : His father Patrick Byrne was an MP, TD, Senator and Lord Mayor of Dublin. Another brother Alfie Byrne was also a TD.
Truth = His father , Patrick Byrne , was a delegate , senator and lord mayor of Dublin , another brother Alfie Byrne was also TD .

Text  > Holly was musically influenced by Elton John .
Pred  < Paraphrase : Holly was musically influenced by Elton John.
Truth = Holly Holly was influenced by Elton John musically .

Text  > Fishman holds a bachelor 's degree from Columbia University and a master 's degree in economics from Brown University .
Pred  < Paraphrase : Fishman holds a bachelor's degree from Columbia University and a master's in economics from Brown University.
Truth = Fishman holds a Bachelor 's degree from Columbia University and a Master 's degree in Brown University economics .

Text  > An analysis comparing millions of RSA public keys gathered from the Internet was announced in 2012 by Lenstra , Hughes , Augier , Bos , Kleinjung , and Wachter .
Pred  < Paraphrase : An analysis of millions of RSA public keys gathered from the Internet was announced by Lenstra, Hughes, Augier, Bos, Kleinjung and Wachter in 2012.
Truth = An analysis of millions of public RSA keys from the Internet was announced in 2012 by Augier , Hughes , Lenstra , Bos , Kleinjung and Wachter .

Text  > According to the United States Census Bureau , Kennesaw is a total area of , of which is land and or 1.08 % , has water .
Pred  < Paraphrase : According to the United States Census Bureau, Kennesaw has a total area of which is land and or 1.08 % has water.
Truth = According to the United States Census Bureau , Kennesaw is a total area of which is land and 1.08 % has water .

Text  > King Qi met with King Zhao of Qin in western Zhou in 284 BC to form an alliance against Xi .
Pred  < Paraphrase : King Qi met in 284 BC with King Zhao of Qin in western Zhou to form an alliance against Xi.
Truth = In 284 BC , King Qi met with King Zhao of Qin in Western Zhou to form an alliance against Xi .

Text  > The Chenzhou Administrative Region in the Tang Dynasty is under the administration of modern Henan in the east of Zhoukou :
Pred  < Paraphrase : The administrative region of Chenzhou in the Tang Dynasty is under the administration of modern Henan in the east of Zhoukou :
Truth = The administrative region of Chenzhou in the Tang dynasty is under the administration of modern Henan in eastern Zhoukou :

Text  > The species was first formally described by the botanist Stephan Endlicher in 1846 as part of Johann Georg Christian Lehmann 's work `` Irideae . Plantae Preissianae '' .
Pred  < Paraphrase : The species was first formally described in 1846 by the botanist Stephan Endlicher as part of Johann Georg Christian Lehmann's work `` Irideae. Plantae Preissianae ''.
Truth = The species was first formally described in 1846 by the botanist Stephan Endlicher as part of the work `` Irideae Plantae Preissianae '' by Johann Georg Christian Lehmann .

Text  > It was designed in 1983 by architects Philip Johnson ( alumnus of the University ) and John Burgee .
Pred  < Paraphrase : It was designed by the architects Philip Johnson ( alumnus of the university ) and John Burgee in 1983.
Truth = It was designed by architects Philip Johnson ( alumnus of the University ) and John Burgee in 1983 .

Text  > This song was written and produced by the gala composed by Filippo Andrea Carmeni and Maurizio Molella .
Pred  < Paraphrase : This song was written and produced by the gala composed by Filippo Andrea Carmeni and Maurizio Molella.
Truth = The song was written and produced by Gala composed by Filippo Andrea Carmeni and Maurizio Molella .

Text  > From 1863 to 1866 , Dunedin and Suburbs North were a parliamentary electorate in the city of Dunedin , Otago , New Zealand , and was a multi-member electorate .
Pred  < Paraphrase : From 1863 to 1866, Dunedin and Suburbs North was a parliamentary electorate in the city of Dunedin, Otago, New Zealand, and was a multi-member electorate.
Truth = Dunedin and Suburbs North was a parliamentary electorate in the city of Dunedin in Otago , New Zealand from 1863 to 1866 . It was a multi-member electorate .

Text  > Kudlago Island is one of the many uninhabited Canadian arctic islands in Qikiqtaaluk Region , Nunavut .
Pred  < Paraphrase : Kudlago Island is one of the many uninhabited Canadian arctic islands in the Qikiqtaaluk region of Nunavut.
Truth = Kudlago Island is one of the many uninhabited Canadian islands in Qikiqtaaluk Region , Nunavut .

Text  > The Jiul de Vest river is a tributary of the River Jidanul in Romania .
Pred  < The Jiul de Vest River is a tributary of the Jidanul River in Romania.
Truth = The Jiul de Vest River is a tributary of the Jidanul River in Romania .

Text  > Tipico Co. Ltd and Tipico Casino Ltd were founded in 2004 as international trading companies in the commercial register of the Malta Financial Services Authority .
Pred  < Paraphrase : Tipico Co. Ltd and Tipico Casino Ltd were founded in 2004 as international trading companies in the commercial register of the Malta Financial Services Authority.
Truth = Tipico Co. Ltd and Tipico Casino Ltd were established in 2004 as international trading companies in the Commercial Register of the Malta Financial Services Authority .

Text  > Lovey and Dude Romeo from Pittsburgh Pennsylvania have published extensively online and in YouTube - Videos Puli PrayingOriginally by Green Bay Wisconsin .
Pred  < Paraphrase : Lovey and Dude Romeo from Pittsburgh Pennsylvania have published extensively online and in YouTube videos Puli Praying originally by Green Bay Wisconsin.
Truth = Lovey and Dude Romeo of Pittsburgh Pennsylvania have appeared extensively online and in YouTube videos Puli PrayingOriginally from Green Bay Wisconsin .

Text  > Tim Tim Henman won against Pete Sampras in the final 6 -- 7 , 6 -- 4 , 7 -- 6 .
Pred  < Tim Henman won against Pete Sampras in the final 6 -- 7, 6 -- 4, 7 -- 6.
Truth = Tim Henman won in the final 6 -- 7 , 6 -- 4 , 7 -- 6 against Pete Sampras .

Text  > Katz was born in Sweden in 1947 and moved to New York City at the age of 1 .
Pred  < Paraphrase : Katz was born in 1947 in Sweden and moved to New York City at the age of 1.
Truth = Katz was born in 1947 in Sweden and moved to New York at the age of one .

Text  > Most releases of the album outside of North America had the same audio content , but located the track markers differently depending on which label released the CD .
Pred  < Paraphrase : Most releases of the album outside of North America had the same audio content, but located the track markers differently depending on which label released the CD.
Truth = Most of the releases of the album outside North America had the same audio content , but the track markers located differently depending on the label that the CD had released .

Text  > The album was recorded in Los Angeles by Aníbal Kerpel and mixed in `` La Casa '' studies in Los Angeles , California .
Pred  < Paraphrase : The album was recorded in Los Angeles by Aníbal Kerpel and mixed in `` La Casa '' studies in Los Los Angeles, California.
Truth = The album was recorded in Los Angeles by Aníbal Kerpel . Mixed in `` La Casa '' studies in Los Angeles , California .

Text  > The crane was used to unload the lorries and to launch the craft after completion , a total of over 2000 lifts in three years .
Pred  < Paraphrase : The crane was used to unload the lorries and to launch the craft after completion, a total of over 2000 lifts in three years.
Truth = The crane was used to unload the trucks and to launch the ship after completion , a total of over 2000 lifts in three years .

Text  > It is limited to three very native areas in Santa Cruz , Monterey Peninsula and San Luis Obispo Counties .
Pred  < Paraphrase : It is limited to three very native areas in Santa Cruz, Monterey Peninsula and San Luis Obispo counties.
Truth = It is limited to three very native areas located in Santa Cruz , Monterey Peninsula , and San Luis Obispo Counties .

Text  > The event was attended by Boutique @ hs Ambassador Kyly Clarke , Australian actress Teresa Palmer and Michala Banas .
Pred  < The event was attended by Boutique @ hs ambassador Kyly Clarke, Australian actress Teresa Palmer and Michala Banas.
Truth = The event was visited by Boutique @ hs - Ambassador Kyly Clarke , Australian actress Teresa Palmer and Michala Banas .

Text  > Although rugby union in Croatia was the main centre for the sport in the former Yugoslavia , there was still quite a bit of rugby played in Slovenia .
Pred  < Although rugby union in Croatia was the main centre for the sport in the former Yugoslavia, there was still quite a bit of rugby played in Slovenia.
Truth = Although the Rugby Union in Croatia was the main centre for sports in the former Yugoslavia , there was still a lot of rugby played in Slovenia .

Text  > ACVM is based in Glasgow and has offices in Edinburgh , Aberdeen , Newcastle , Manchester and Milton Keynes .
Pred  < ACVM is based in Glasgow and has offices in Edinburgh, Aberdeen, Newcastle, Manchester and Milton Keynes.
Truth = ACVM is based in Glasgow and has subsidiaries in Edinburgh , Aberdeen , Newcastle , Manchester and Milton Keynes .

Text  > On September 14 , 2006 , Lang was signed by the Washington Wizards and released by the Wizards in July 2017 .
Pred  < Paraphrase : Lang was signed by the Washington Wizards on 14 September 2006 and released by the Wizards in July 2017.
Truth = On September 14 , 2006 , Lang was signed by the Washington Wizards . In July 2017 , Lang was released by the Wizards .

Text  > Vermont South is bordered by Mitcham to the north , Nunawading and Forest Hill to the west , Vermont to the south and Wantirna and Ringwood to the east .
Pred  < Paraphrase : Vermont South is bordered by Mitcham to the north, Nunawading and Forest Hill to the west, Vermont to the south and Wantirna and Ringwood to the east.
Truth = Vermont South is bordered to the north of Mitcham , to the west by Nunawading and Forest Hill , to the south by Vermont and to the east by Wantirna and Ringwood .

Text  > It is also worth noting that the following code would work without ADL ( it will be applied to it anyway ) .
Pred  < It is also worth noting that the following code would work without ADL ( it will be applied to it anyway ).
Truth = It is also worth noting that the following code would work without ADL ( it 's applied to it anyway ) .

Text  > It is found in North America , where it was recorded from Newfoundland and Labrador west to British Columbia , north to Alaska and the Yukon .
Pred  < Paraphrase : It is found in North America, where it was recorded from Newfoundland and Labrador west to British Columbia, north to Alaska and the Yukon.
Truth = It is found in North America , where it has been recorded from Newfoundland and Labrador west to British Columbia , north to Alaska and the Yukon .

Text  > In 1915 bus service began between Overlea and Belair `` Jitney buses '' , which operated for several years .
Pred  < Paraphrase : In 1915, bus service began between Overlea and Belair `` Jitney buses '', which operated for several years.
Truth = In 1915 , bus service called between Overlea and Belair began `` jitney buses . '' These operated for several years .

Text  > The species are members of various ecological groups , including tropical shrubs , lianas and trees , xerophytic plants , mycoheterotrophs , as well as different herbaceous representatives .
Pred  < Paraphrase : The species are members of various ecological groups, including tropical shrubs, lianas and trees, xerophytic plants, mycoheterotrophs as well as different herbaceous representatives.
Truth = The species are members of different ecological groups , including tropical shrubs , lianas and trees , xerophytic plants , mycoheterotrophic as well as various herbal representatives .

```

 ---
 
 ## Refrences

- [BERT NLP — How To Build a Question Answering Bot](https://towardsdatascience.com/bert-nlp-how-to-build-a-question-answering-bot-98b1d1594d7b)
- [SQUAD DATASET](https://rajpurkar.github.io/SQuAD-explorer/)
- [BART for Paraphrasing with Simple Transformers](https://towardsdatascience.com/bart-for-paraphrasing-with-simple-transformers-7c9ea3dfdd8c)
- [BART Simple Transformers Github](https://github.com/ThilinaRajapakse/simpletransformers/tree/master/examples/seq2seq/paraphrasing)
- [Pytorch Hugging Face](https://pytorch.org/hub/huggingface_pytorch-transformers/)
- [BERT Fine-Tuning Tutorial with PyTorch](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
