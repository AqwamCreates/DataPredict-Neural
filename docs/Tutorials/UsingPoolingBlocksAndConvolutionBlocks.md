# Using Pooling Blocks And Convolution Blocks

Pooling blocks and convolution blocks allow use to capture spatial information from the data. These blocks can be confusing to use at first, but in this tutorial, we will show you how those blocks work.

## Requirements

* An understanding on these two tutorials:

  * [General Tensor Conventions](GeneralTensorConventions.md)

  * [Spatial Dimension, Kernel And Stride](SpatialDimensionKernelAndStride.md)

## The Pooling Blocks

We will first create our inputTensor and the average pooling block object as shown below for the purpose of this tutorial.

```lua

local inputTensor = TensorL:createRandomNormalTensor({10, 10, 10, 10}) -- Creating a 4D tensor with the size of 10 x 10 x 10 x 10.

local AveragePooling2D = DataPredict.PoolingBlocks.AveragePooling2D.new({kernelDimensionSizeArray = {2, 2}, strideDimensionSizeArray = {2, 2})

```

In here, we can see that we have created a 4D tensor. This is because:

* The first dimension is used for the number of data.

* The second dimension is used for the number of kernels.

The last two dimension is the kernel dimensions, where the average pooling block requires these dimensions to get the average input value for the output value.
