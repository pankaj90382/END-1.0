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


#### BERT

**B**idirectional **E**ncoder **R**epresentations from **T**ransformers

BERT is basically a trained Transformer Encoder stack. Both BERT models (Large & Base) have large number of encoder layers (Transformer Blocks) - 12 for the Base version and 24 for the large version.

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

BART is a denoising autoencoder built with a sequence-to-sequence model that is applicable to a very wide range of end tasks.

### Pretraining: Fill In the Span

BART is trained on tasks where spans of text are replaced by masked tokens, and the model must learn to reconstruct the original document from this altered span of text.

BART improves on BERT by replacing the BERT's fill-in-the-blank cloze task with a more complicated mix of pretraining tasks.

![text infilling](https://miro.medium.com/max/700/1*jWecxbzBsJEbNgiy_LOIvw.png?raw=true)

In the above example the origin text is ` A B C D E` and the span `C, D` is masked before sending it to the encoder, also an extra mask is placed between `A` and `B` and one mask is removed between `B` and `E`, now the corrupted document is `A _ B _ E`. The encoder takes this as input, encodes it and throws it to the decoder.

The decoder must now use this encoding to reconstruct the original document. `A B C D E`

**Training Logs**


**Sample Outputs**

 ---
 
 ## Refrences

- [BERT NLP — How To Build a Question Answering Bot](https://towardsdatascience.com/bert-nlp-how-to-build-a-question-answering-bot-98b1d1594d7b)
- [SQUAD DATASET](https://rajpurkar.github.io/SQuAD-explorer/)
- [BART for Paraphrasing with Simple Transformers](https://towardsdatascience.com/bart-for-paraphrasing-with-simple-transformers-7c9ea3dfdd8c)
- [BART Simple Transformers Github](https://github.com/ThilinaRajapakse/simpletransformers/tree/master/examples/seq2seq/paraphrasing)
- [Pytorch Hugging Face](https://pytorch.org/hub/huggingface_pytorch-transformers/)
- [BERT Fine-Tuning Tutorial with PyTorch](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
