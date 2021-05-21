# Session 3 - MNIST ADDER NETWEOK DESIGN

## Objective
1.  Write a neural network that can:
    1.  take 2 inputs:
        1.  an image from MNIST dataset, and
        2.  a random number between 0 and 9
    2.  and gives two outputs:
        1.  the "number" that was represented by the MNIST image, and
        2.  the "sum" of this number with the random number that was generated and sent as the input to the network
    3.  you can mix fully connected layers and convolution layers
    4.  you can use one-hot encoding to represent the random number input as well as the "summed" output.
2.  Your code MUST be:
    1.  well documented (via readme file on github and comments in the code)
    2.  must mention the data representation
    3.  must mention your data generation strategy
    4.  must mention how you have combined the two inputs
    5.  must mention how you are evaluating your results
    6.  must mention "what" results you finally got and how did you evaluate your results
    7.  must mention what loss function you picked and why!
    8.  training MUST happen on the GPU

## Solution

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pankaj90382/END-1.0/blob/main/S3/MNIST_ADDER.ipynb)

### Data Representation

The Data sample is always represented as `(input, target)` where both the input and target have to be `torch.Tensor`, conversion to Tensor is taken care by PyTorch's `DataLoader` for primitive types, for MNIST Image we need to use the torchvision transforms' `ToTensor` method.

My Representation for `(img, target` is `(img, random_number),  (target, target + random_number)`

### Data Generation Strategy

```python
class MyDataset(MNIST):

    def __init__(self, *args, **kwargs):
        super(MyDataset, self).__init__(*args, **kwargs)
        
    def __getitem__(self, index):
        img, target = super(self.__class__, self).__getitem__(index)
        random_number = np.random.randint(low=0, high=10)
        return (img, random_number), (target, target + random_number)
```
### The Network

The model's primary task is to figure out the MNIST Image Classification, for that I simply copy-paste one of my past MNIST Classification Model and only use the backbone layers, which is upto the GAP Layer + Conv1D,

Now after this i got 20 Channels, there 20 channels are flattened out, and concatenated with the one-hot representation of the "random" number, so 30 features in total, Now from here on i use Linear Layers which will do the addition, and out of that I will extract 10 features of my MNIST classification, and 19 features of my Addition classification.

Why 19 features ? because 0-9 (MNIST) + 0-9 (Random) will be 0-18 numbers, or 19 possible numbers in total.

```python
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1) #input -? OUtput? RF
        self.conv2 = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1),nn.BatchNorm2d(8))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv4 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1),nn.BatchNorm2d(16),nn.Dropout(0.23))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(16, 32, 3)
        self.conv6 = nn.Conv2d(32, 32, 3)
        self.conv7 = nn.Conv2d(32, 10, 3)
        self.adder_layer1 = nn.Sequential(nn.Linear(in_features=20, out_features=60, bias=False),nn.BatchNorm1d(60),nn.ReLU(),nn.Dropout(0.1))
        self.adder_layer2 = nn.Sequential(nn.Linear(in_features=60, out_features=60, bias=False),nn.BatchNorm1d(60),nn.ReLU(),nn.Dropout(0.1))
        self.adder_layer3 = nn.Sequential(nn.Linear(in_features=60, out_features=19, bias=False))


    def forward(self, x, y):
        y = F.one_hot(y, num_classes=self.num_classes)
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = self.conv7(x)
        x = x.view(-1, 10)
        y = torch.cat([x, y], dim=-1)
        y = self.adder_layer1(y)
        y = self.adder_layer2(y)
        y = self.adder_layer3(y)
        return F.log_softmax(x), F.log_softmax(y)
```

### Loss Functions

The first task is to classify the mnist image, so choice of loss function for that is negative log likelihood loss` (negative log likelihood).

But for the adder, since i used a one hot representation for the output, negative log likelihood loss seems like a good choice for it too. So i went with that.

Now both these losses are combined by simple addition. We can also give more weightage to the MNIST Loss, because without the correct prediction for MNIST we cannot give the correct output for the adder.

```python
        
        ......
        both mnist and adder_loss use negative log likelihood loss
        mnist_pred, final_pred = model(mnist_x, rand_num)
        mnist_loss = F.nll_loss(mnist_pred, mnist_y)
        adder_loss = F.nll_loss(final_pred, final_y)
```

### Results Evaluation

The Output obtained from the model is evaluated against the target.

```python

            .......
            mnist_pred = torch.argmax(mnist_pred, dim=1)
            final_pred = torch.argmax(final_pred, dim=1)
            mnist_correct += mnist_pred.eq(mnist_y.view_as(mnist_pred)).sum().item()
            adder_correct += final_pred.eq(final_y.view_as(final_pred)).sum().item()
```

Accuracy of MNIST After 20 Epochs:- `99.33`
Accuracy of ADDER After 20 Epochs:- `99.21`


### Sample Test Outputs
|  ![Results_1]()  |  ![Results_2]() |


## Training Logs

The Model was trained for `20` epochs with Optimizer `SGD: Stochastic Gradient Descent`.
