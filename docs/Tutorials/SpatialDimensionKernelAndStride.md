# Spatial Dimension, Kernel And Stride

From the previous [tutorial](GeneralTensorConventions.md), we can see that the dimension size array holds a number of important information. However, these are not enough to fully understand on how to use them when spatial blocks are involved. 

Here are the list of spatial blocks that are available in this library:

* Convolution blocks

* Pooling blocks

In this tutorial, I will further explain what does the "spatial" mean and how these spatial function blocks affects the dimension size array of any input tensors and output tensors.

## The Spatial Dimension

"Spatial" refers to anything related to space or the arrangement of objects in space. It can refer to aspects like time, length, height and width.

When people say 1D max pooling, it just generally means that the max pooling is applied along the length or time. For 2D max pooling, it means that the max pooling is applied along the length and height, or time and length.

Now remember the general tensor conventions that was from the previous tutorial, where each of the dimensions holds specific values for the input tensor:

| Dimension | Meaning                                   |
|-----------|-------------------------------------------|
| 1         | Number of data                            |
| 2         | Number of channels                        |
| N + 2     | Number of width, height, length and so on |

The N + 2 dimensions can be also referred as the spatial dimensions.

## The Kernel

Since we have established that the spatial dimensions are located at N + 2 dimension, we can now understand how kernels are applied.

### The Kernel Dimension Size Array

You may already have seen that the convolutional blocks and pooling blocks contains the 1D, 2D and 3D. Those N-D dimensions refers to the spatial dimensions. Hence, the input tensor's dimension size array must contain those spatial dimensions. For example:

* 1D for data + 1D for channel + 1D for spatial = 3D tensor

* 1D for data + 1D for channel + 2D for spatial = 4D tensor

* 1D for data + 1D for channel + 3D for spatial = 5D tensor

Now, you understand why the convolution blocks and pooling blocks generates and error when you supply them an input tensor that has incorrect number of dimensions.

### The Number Of Kernels

The convolution blocks have "numberOfKernels" as one of its parameters. This determines the number of channels that will be produced for the output tensor, regardless of the number of channels from the input tensor. So, if we have 3 kernels, then it will produce an output tensor that has 3 channels. Pretty simple, right?

## The Stride

Stride just refers to how much the kernels should move along the input tensor's spatial dimension. So let's say we have this example that is shown below:

```lua

local strideDimensionSizeArray = {3, 9, 4}

```

Basically this means that the kernel moves the size of:

*  Three for dimension 1

*  Nine for dimension 2

*  Four for dimension 3

Once we have all these knowledge, we can now calculate the output size for a given dimension. In general the output size can be calculated as:

```lua

local outputSize = ((inputSize - kernelSize) / strideSize) + 1

```

That is all for this tutorial. I do hope you understand what the spatial dimensions are and why the spatial blocks requires specific number of dimensions for our input tensor. 

Now, go play around with the spatial blocks since you now have this knowledge.
