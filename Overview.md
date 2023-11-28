# Pytorch Overview
## Table of Contents
1. [What is Pytorch](#what-is-pytorch) 
    1. [Why use Pytorch](#why-use-pytorch)
    2. [How to used Pytorch](#how-to-used-pytorch)
2. [What can Pytorch do](#what-can-pytorch-do)
3. [Other Questions](#other-questions)
    1. [When was Pytorch created](#when-was-pytorch-created)
    2. [Why I selected Pytorch](#why-i-selected-pytorch)
    3. [How it shaped how I learn python](#how-it-shaped-how-i-learn-python)
    4. [My overall experience with Pytorch](#my-overall-experience-with-pytorch)
***
### What is Pytorch

Pytorch is a...

>"END-TO-END MACHINE LEARNING FRAMEWORK"
>
> PyTorch enables fast, flexible experimentation and efficient production through a user-friendly front-end, distributed training, and ecosystem of tools and libraries." [https://pytorch.org/features/]

since it is used in machine learning it has powerful tensor functions and a matrix is a rank 2 or 2nd order tensor. So I used it in my textual calculator I made for the previous exploration assignment, so that I could easily implement matrix math on top of what I already had.

#### Why use Pytorch

The purpose behind using Pytorch tensors instead of Numpy ndarray is that Pytorch's tensors are more efficient and stable when it comes down to tensor math operations and most tensor functions. Another reason why I used Pytorch instead is because I already know a lot about numpy ndarray and I want to explore posible alternatives.

#### How to used Pytorch

Here's a basic guide on how to use Pytorch in Python:

##### Install Pytorch

To install Pytorch you will need to go to their website [https://pytorch.org/get-started/locally/] on this web page you can input what kind of Pytorch build you want (I used the Stable (2.1.1) build), what your operating system is, what package you use to install python libraries (I used Pip), what language you want to use (Obviously I used python), and what computation platform you want to use (I used CUDA 11.8). After doing all of that it will give you a command you will need to run in your console, however it is a very big library so it likely won't be able to be installed on a lab computer.

##### Import the torch Module:

To use Pytorch in Python, you need to import the torch module. You can do this with the following line of code:

```py
import torch
```
##### Creating a tensor:

Before being able to access any of the math functions and/or other functionallities you need to make a tensor. To make a tensor use the `torch.Tensor()` or `torch.tensor()` function (from what I found these have the same functionallity):

```py
tensor1 = torch.Tensor([an array of either numbers or more arrays])
```
Replace 'an array of either numbers or more arrays' with your values you want in a tensor.

for example...

```py
>>> tensor1 = torch.Tensor([[1,2,3],[4,5,6],[7,8,9]])
>>> print(tensor1)
tensor([[1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]])
```

##### Using Pytorch tensor functions:

Once you have a tensor, you can use it for various purposes. Here are some common tensor functions in Pytorch:

`is_tensor()`: This function determines if its input is a tensor.

```py
>>> print(torch.is_tensor(tensor1))
True
>>> print(torch.is_tensor(123))
False
```

`numel()`: This function returns the number of elements in its input.

```py
>>> print(torch.numel(tensor1))
9
```

`zeros()`: Creates a tensor with the size of input (which is list)

```py
>>> print(torch.zeros([2,2]))
tensor([[0,0],
        [0,0]])
```

`transpose()`: Takes a tensor, and two ints representing which dimensions are going to be swaped. (if the tensor isn't a sparse tensor, the transposed tensor uses the same storage) 

```py
>>> print(torch.transpose(tensor1, 1, 0))
tensor([[1., 4., 7.],
        [2., 5., 8.],
        [3., 6., 9.]])
```

##### Tensor Math:

`add()`: this takes a tensor and either a number or another tensor, if given a number it adds it to all elements in the tensor and if given a tensor it adds elements who shares their indexes.

similarly: `mul()`, `div()`, `sub()`, `pow()`

```py
>>> print(torch.add(tensor1, 1))
tensor([[2., 3., 4.],
        [5., 6., 7.],
        [8., 9., 10.]])
>>> print(torch.sub(tensor1, 1))
tensor([[0., 1., 2.],
        [3., 4., 5.],
        [6., 7., 8.]])
...
```

`det()`: calculates the determinant of the input tensor

`trace()`: calculates the trace of the input tensor (which is the sum of the diagonal)

```py
>>> print(torch.det(tensor1))
0
>>> print(torch.trace(tensor1))
15
```

##### Accessing Elements:
Pytorch use the same notation as regular arrays so you can access elements with sets of [ ] however you will need to use `.item()` if you want to get the specific value.

```py
>>> print(tensor1[2][2])
tensor(9.)
>>> print(tensor1[2][2]).item()
9.0
```

***

### What can Pytorch do

PyTorch as mentioned is a powerful deep learning framework with a wide range of functionalities, and the following makes it popular among researchers and practitioners. Here's a list of some key functionalities of PyTorch:

1. **Tensors**: PyTorch provides a multi-dimensional array called a tensor, similar to NumPy ndarrays.

    Tensors can be used for numerical computations and are the fundamental data structure in PyTorch. 
    
    I used these in my sample program to do matrix/vector math.
    a sample can be seen in the above section

2. **Autograd**: Automatic differentiation through the autograd system allows for computing gradients of tensors.
    
    Gradients are crucial for training neural networks using gradient-based optimization algorithms.

    ```py
    # Create a tensor with requires_grad=True
    x = torch.tensor([2.0], requires_grad=True)

    # Perform operations
    y = x ** 2
    z = y.mean()

    # Compute gradients
    z.backward()

    # Access gradients
    print(x.grad)

    # OUTPUT: tensor([4.])
    ```

3. **Dynamic Computational Graph**: PyTorch uses a dynamic computational graph, allowing for dynamic graph creation during runtime.
    
    This dynamic nature is beneficial for implementing dynamic architectures, like recurrent neural networks (RNNs).

    ```py
    # Dynamic graph creation during runtime
    def dynamic_model(x):
        if x.sum() > 0:
            return x * 2
        else:
            return x * 3

    input_data = torch.tensor([1, 2, 3])

    output = dynamic_model(input_data)
    # output: tensor([2,4,6])
    ```

4. **Neural Network Building Blocks**: The torch.nn module provides a set of building blocks for constructing neural networks, including layers, activation functions, loss functions, and optimization algorithms.

    ```py
    import torch.nn as nn

    # Define a simple neural network
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc = nn.Linear(10, 5)

        def forward(self, x):
            x = self.fc(x)
            return x

    model = SimpleNet()
    # model: 
    # SimpleNet(
    #   (fc): Linear(in_features=10, out_features=5, bias=True)
    # )
    ```

5. **Optimizers**: PyTorch includes a variety of optimization algorithms in the torch.optim module, such as Stochastic Gradient Descent (SGD), Adam, and RMSprop.

    Optimizers facilitate the training of neural networks by updating model parameters based on computed gradients.

    ```py
    import torch.optim as optim

    # Instantiate a model and an optimizer
    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Inside training loop
    optimizer.zero_grad()
    output = model(input_data)
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    optimizer.step()
    ```

6. **Data Loading and Processing**: The torch.utils.data module provides tools for efficient data loading and processing.

    Custom datasets and data loaders can be created to handle large datasets and provide batching during training.

    ```py
    from torch.utils.data import Dataset, DataLoader

    # Define a custom dataset
    class CustomDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    # Create a data loader
    dataset = CustomDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    ```

7. **GPU Acceleration**: PyTorch seamlessly supports GPU acceleration, allowing users to perform computations on CUDA-enabled GPUs.
    
    Tensor operations and model parameters can be easily moved between CPU and GPU using the `.to()` method.

    ```py
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move tensors and models to GPU
    tensor_on_gpu = tensor_a.to(device)
    model_on_gpu = SimpleNet().to(device)
    ```

8. **Model Saving and Loading**: PyTorch allows users to save and load model checkpoints using the torch.save() and torch.load() functions.

    This functionality is crucial for saving trained models and resuming training or making predictions later.

    ```py
    # Save model checkpoint
    torch.save(model.state_dict(), 'model_checkpoint.pth')

    # Load model checkpoint
    model = SimpleNet()
    model.load_state_dict(torch.load('model_checkpoint.pth'))
    ```

>NOTE: all code examples are simplified ideas of what I learned from looking through the examples from the Pytorch website [https://pytorch.org/examples/?utm_source=examples&utm_medium=examples-landing] and from videos I've watched over the last month like [https://www.youtube.com/watch?time_continue=1&v=u7x8RXwLKcA&embeds_referring_euri=https%3A%2F%2Fpytorch.org%2F&source_ve_path=Mjg2NjY&feature=emb_logo]

These functionalities collectively cover the core aspects of using PyTorch for deep learning, from tensor manipulation and model building to optimization and deployment. They form the foundation for a wide range of applications in research and production environments. 

As I have stated this can do much more than what I needed, but I have explored a lot of what it has to offer, outside of my sample program.  
***
<div style="page-break-before:always">&nbsp;</div>
<p></p>

### Other Questions

#### When was Pytorch created

Pytorch was create in september 2016, it was made by a team from Facebook's research lab. [https://datascientest.com/en/pytorch-all-about-this-framework]

#### Why I selected Pytorch

I selected Pytorch as I already knew a lot about numpy's ndarray, but I still wanted to implement matrix math into my textual calculator, as well I want to learn the basics of Pytorch as I find neural networks very interesting and this was a good first step in learning how to make a neural network and other machine learning topics.

#### How it shaped how I learn python

I have learned a lot about how useful dictionaries can be when they contain function references and also how useful lambda functions can be. This has shown me a easy way to solve problems that I didn't know how to solve before. 

#### My overall experience with Pytorch

My overall experience with Pytorch is very good as I was able to expand upon a program that I have been wanting to make for a long time.

I would recomend Pytorch to anyone who needs to make a neural network and/or make machine learning, and it is useful for simple matrix math, but numpy is likely better for just matrix math as the pytorch library slows down how fast my program starts up which isn't really a bad thing especially if you are using most of the neural network / machine learning aspect but if you are just using it for tensor math it would be better to use numpy as its more specific to math. 

I might use it in the future if I decide to make a neural network, but for math I'll stick to numpy.