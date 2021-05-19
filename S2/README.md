# Session 2 - Neural Network Design (BackProp)

## Objective:

Create excel sheet showing backpropagation. Explain each major step with following guidelines. 
- Use exactly the same values for all variables as used in the class
- Take a screenshot, and show that screenshot in the readme file
- Excel file must be there for us to cross-check the image shown on readme (no image = no score)
- Explain each major step
- Show what happens to the error graph when you change the learning rate from [0.1, 0.2, 0.5, 0.8, 1.0, 2.0]

## Solution
[![Excel](https://shields.io/badge/-Download-217346?logo=microsoft-excel&style=flat)](https://github.com/pankaj90382/END-1.0/raw/main/S2/NN_Design.xlsx)
### Loss vs Iteration Graph with respect to Learning Rate

The graph shows how the learning rate changes the whole game in neural network. 
1.  When the **learning rate is 0.1** the overall loss decreases slowly compared to other learning rates. It needs high number of epochs to reach the accuracy. 
2.  As **learning rate varies from 0.1 to 1.0**, the overall loss of the neural network dimnishes very rapidly on every epoch. The learning rate below 0.5 creates a decreasing linear graph, howerver, on other side it creates decreasing exponential graph.
3.  In the case the **learning rate from 1.0 to 2.0**, after 10th to 15th epoch, the overall loss reaches to point where it cannot be dimnishes further and creating a constantgraph.

![Loss vs Iteration](LR.JPG)

### Learning Rate with 0.5 values

That's interesting to look how the total loss values rapidly dimnishes with constant learning rate. Every line considered as one epoch and all the weights are trained according to the backpropagation with aspect to learning rate (Explainrd Later in Mathematics).

![Learning Rate (0.5)](LR-0.5.JPG)

### Steps Explnation (Mostly Math Derivative Part)

![Neural Network Design](NN.jpg)

To train above neural network model, there are two most basic things which are loss and learning rate. To decrease the loss, one popular method used is gradient descent because it creates concave loss function graph, it's best possible to chance to avoid local minima. The gradient loss function can be defined by

<h3 align="center"> w<sub>i</sub><sup>new</sup> = w<sub>i</sub><sup>old</sup> - (learning_rate * &part;E<sub>total</sub>/&part;w<sub>i</sub>) </h3>

where
*  w<sub>i</sub><sup>new</sup> = new weights
*  w<sub>i</sub><sup>old</sup> = old weights
*  &part;E<sub>total</sub>/&part;w<sub>i</sub> = error total with respect to weights.

Now we have weights W = [w<sub>1</sub>, w<sub>2</sub>, w<sub>2</sub>, w<sub>3</sub>, w<sub>4</sub>, w<sub>5</sub>, w<sub>6</sub>, w<sub>7</sub>, w<sub>8</sub>] are the weights used

and

&part;E<sub>total</sub>/&part;w<sub>i</sub> = [ &part;E<sub>total</sub>/&part;w<sub>1</sub>, &part;E<sub>total</sub>/&part;w<sub>2</sub>, &part;E<sub>total</sub>/&part;w<sub>3</sub>, &part;E<sub>total</sub>/&part;w<sub>4</sub>, &part;E<sub>total</sub>/&part;w<sub>5</sub>, &part;E<sub>total</sub>/&part;w<sub>6</sub>, &part;E<sub>total</sub>/&part;w<sub>7</sub>, &part;E<sub>total</sub>/&part;w<sub>8</sub>]

Now see carefully, the below equations, along with our network diagram above somewhere, very important!

h<sub>1</sub> = w<sub>1</sub>*i<sub>1</sub> + w<sub>2</sub>*i<sub>2</sub> <br>

h<sub>2</sub> = w<sub>3</sub>*i<sub>1</sub> + w<sub>4</sub>*i<sub>2</sub> <br><br>

a<sub>h1</sub> = &sigma;(h<sub>1</sub>) <br>

a<sub>h2</sub> = &sigma;(h<sub>2</sub>) <br><br>


o<sub>1</sub> = w<sub>5</sub>*a<sub>h1</sub> + w<sub>6</sub>*a<sub>h2</sub> <br>

o<sub>2</sub> = w<sub>7</sub>*a<sub>h1</sub> + w<sub>8</sub>*a<sub>h2</sub> <br>



a<sub>o1</sub> = &sigma;(o<sub>1</sub>) <br>

a<sub>o2</sub> = &sigma;(o<sub>2</sub>) <br><br>



E<sub>1</sub> = (1/2) *(t<sub>1</sub>-a<sub>o1</sub>)<sup>2</sup>  <br>
E<sub>2</sub> = (1/2) *(t<sub>2</sub>-a<sub>o2</sub>)<sup>2</sup>  <br>


Where
        [i<sub>1</sub>, i<sub>2</sub>] and [t<sub>1</sub>, t<sub>2</sub>] are the inputs and target outputs respectively

Now we can start to compute the partial derivatives w.r.t to the weights, and also remember to use chain rule wherever applicable and necessary.

&part;E<sub>total</sub>/&part;w<sub>5</sub> = &nbsp;&part;E<sub>1</sub>/&part;w<sub>5</sub>&nbsp;
=  &nbsp;&part;E<sub>1</sub>/&part;a<sub>o1</sub> * &part;a<sub>o1</sub>/&part;o<sub>1</sub> *  &part;o<sub>1</sub>/&part;w<sub>5</sub>  <br>

We didn't consider E2 above, because it does no contribution to w<sub>5</sub>

&part;E<sub>1</sub>/&part;a<sub>o1</sub> = &nbsp;&part;((1/2) *(t<sub>1</sub>-a<sub>o1</sub>)<sup>2</sup>)/&part;a<sub>o1</sub> = a<sub>o1</sub> - t<sub>1</sub>

&part;a<sub>o1</sub>/&part;o<sub>1</sub> = &nbsp;&part;&sigma;(o<sub>1</sub>)/&part;o<sub>1</sub> = &nbsp;&sigma;(o<sub>1</sub>)(1-&sigma;(o<sub>1</sub>) = a<sub>o1</sub>*(1-a<sub>o1</sub>)<br>

and <br>

&part;o<sub>1</sub>/&part;w<sub>5</sub> = a<sub>h1</sub> <br>

similarly we can complete it for other weights in the last hidden layer

&part;E<sub>total</sub>/&part;w<sub>5</sub> =  (a<sub>o1</sub>-t<sub>1</sub>) * a<sub>o1</sub>*(1-a<sub>o1</sub>) * a<sub>h1</sub> <br>
&part;E<sub>total</sub>/&part;w<sub>6</sub> =  (a<sub>o1</sub>-t<sub>1</sub>) * a<sub>o1</sub>*(1-a<sub>o1</sub>) * a<sub>h2</sub> <br>
&part;E<sub>total</sub>/&part;w<sub>7</sub> =  (a<sub>o2</sub>-t<sub>2</sub>) * a<sub>o2</sub>*(1-a<sub>o2</sub>) * a<sub>h1</sub> <br>
&part;E<sub>total</sub>/&part;w<sub>8</sub> =  (a<sub>o2</sub>-t<sub>2</sub>) * a<sub>o2</sub>*(1-a<sub>o2</sub>) * a<sub>h2</sub> <br>

Now coming to the first hidden layer weights

&part;E<sub>total</sub>/&part;w<sub>1</sub> = &part;E<sub>total</sub>/&part;a<sub>o1</sub> * &part;a<sub>o1</sub>/&part;o<sub>1</sub> * &part;o<sub>1</sub>/&part;a<sub>h1</sub> *  &part;a<sub>h1</sub>/&part;h<sub>1</sub> *  &part;h<sub>1</sub>/&part;w<sub>1</sub> <br>

Seen this somewhere, right, we did this before

&part;E<sub>total</sub>/&part;a<sub>o1</sub> * &part;a<sub>o1</sub>/&part;o<sub>1</sub> *  &part;o<sub>1</sub>/&part;a<sub>h1</sub>
 =  &part;(E<sub>1</sub> + E<sub>2</sub>)/&part;a<sub>h1</sub> =  &nbsp; &part;E<sub>1</sub>/&part;a<sub>h1</sub> * &part;E<sub>2</sub>/&part;a<sub>h1</sub> = &nbsp;((a<sub>o1</sub>-t<sub>1</sub>) * a<sub>o1</sub>*(1-a<sub>o1</sub>) * w<sub>5</sub> + (a<sub>o2</sub>-t<sub>2</sub>) * a<sub>o2</sub>*(1-a<sub>o2</sub>) * w<sub>7</sub>)

Now we can compute all the gradient of the first layer

&part;E<sub>total</sub>/&part;w<sub>1</sub> = &nbsp; &part;E<sub>total</sub>/&part;a<sub>h1</sub> * a<sub>h1</sub>*(1-a<sub>h1</sub>) * i<sub>1</sub> = &nbsp;((a<sub>o1</sub>-t<sub>1</sub>) * a<sub>o1</sub>*(1-a<sub>o1</sub>) * w<sub>5</sub> + (a<sub>o2</sub>-t<sub>2</sub>) * a<sub>o2</sub>*(1-a<sub>o2</sub>) * w<sub>7</sub>) * a<sub>h1</sub>*(1-a<sub>h1</sub>) * i<sub>1</sub> <br>
 &part;E<sub>total</sub>/&part;w<sub>2</sub> = &nbsp; &part;E<sub>total</sub>/&part;a<sub>h1</sub> * a<sub>h1</sub>*(1-a<sub>h1</sub>) * i<sub>2</sub> = &nbsp;((a<sub>o1</sub>-t<sub>1</sub>) * a<sub>o1</sub>*(1-a<sub>o1</sub>) * w<sub>5</sub> + (a<sub>o2</sub>-t<sub>2</sub>) &nbsp;* a<sub>o2</sub>*(1-a<sub>o2</sub>) * w<sub>7</sub>) * a<sub>h1</sub>*(1-a<sub>h1</sub>) * i<sub>2</sub> <br>
 &part;E<sub>total</sub>/&part;w<sub>3</sub> = &nbsp; &part;E<sub>total</sub>/&part;a<sub>h2</sub> * a<sub>h2</sub>*(1-a<sub>h2</sub>) * i<sub>1</sub> = &nbsp;((a<sub>o1</sub>-t<sub>1</sub>) * a<sub>o1</sub>*(1-a<sub>o1</sub>) * w<sub>6</sub> + (a<sub>o2</sub>-t<sub>2</sub>) &nbsp;* a<sub>o2</sub>*(1-a<sub>o2</sub>) * w<sub>8</sub>) * a<sub>h2</sub>*(1-a<sub>h2</sub>) * i<sub>1</sub> <br>
 &part;E<sub>total</sub>/&part;w<sub>4</sub> = &nbsp; &part;E<sub>total</sub>/&part;a<sub>h2</sub> * a<sub>h2</sub>*(1-a<sub>h2</sub>) * i<sub>2</sub> = &nbsp;((a<sub>o1</sub>-t<sub>1</sub>) * a<sub>o1</sub>*(1-a<sub>o1</sub>) * w<sub>6</sub> + (a<sub>o2</sub>-t<sub>2</sub>) * a<sub>o2</sub>*(1-a<sub>o2</sub>) * w<sub>8</sub>) * a<sub>h2</sub>*(1-a<sub>h2</sub>) * i<sub>2</sub>
 
 
