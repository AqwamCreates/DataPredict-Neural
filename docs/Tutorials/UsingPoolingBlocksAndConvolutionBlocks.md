# Using Pooling Blocks And Convolution Blocks

Pooling blocks and convolution blocks allow use to capture spatial information from the data. These blocks can be confusing to use at first, but in this tutorial, we will show you how those blocks works.

## Requirements

* An understanding on these two tutorials:

  * [General Tensor Conventions](GeneralTensorConventions.md)

  * [Spatial Dimension, Kernel And Stride](SpatialDimensionKernelAndStride.md)

## The Pooling Blocks

We will first create our input tensor and the average pooling 2D block object as shown below for the purpose of this tutorial.

```lua

local inputTensor = TensorL:createRandomNormalTensor({20, 3, 10, 10}) -- Creating a 4D tensor with the size of 20 x 3 x 10 x 10.

local AveragePooling2D = DataPredict.PoolingBlocks.AveragePooling2D.new({kernelDimensionSizeArray = {2, 2}, strideDimensionSizeArray = {2, 2})

```

In here, we can see that we have created a 4D tensor. This is because:

* The first dimension is used for the number of data.

* The second dimension is used for the number of channels.

The last two dimensions are used for the kernel dimensions, where the average pooling 2D block requires these dimensions to get the average input value for the output value. Note that if you use average pooling 1D, you only need one kernel dimension.

Once the input tensor and the average pooling 2D block is created, we can transform the input tensor into output tensor.

```lua

local outputTensor = AveragePooling2D:transform(inputTensor)

local outputTensorDimensionSizeArray = TensorL:getDimensionSizeArray(outputTensor)

print(outputTensorDimensionSizeArray) -- This would be a 4D tensor with the size of 20 x 3 x 5 x 5.

```

From here, we can observe that the first two dimension sizes remain the same. This is because the pooling operation generally affects the dimensions after the second one.

## The Convolution Blocks

The convolution blocks generally behaves the same as the pooling blocks. However, the difference is that the convolution blocks will change the number of channels. 

Below, we will on demonstrate how the way we set up the convolution blocks affects the number of channels. Additionally, We will also use the same input tensor that we have used for the pooling block.

```

local Convolution2D = DataPredict.ConvolutionBlocks.Convolution2D.new({numberOfKernels = 7, kernelDimensionSizeArray = {2, 2}, strideDimensionSizeArray = {2, 2})

```

In here, notice that the convolution block has the same parameters as the pooling block, except for the numberOfKernels parameter. The numberOfKernels parameter determines the number of channels for the output tensor.

Below, we will demonstrate on how the input tensor would be changed when it is passed through the convolution block.

```lua

local outputTensor = Convolution2D:transform(inputTensor)

local outputTensorDimensionSizeArray = TensorL:getDimensionSizeArray(outputTensor)

print(outputTensorDimensionSizeArray) -- This would be a 4D tensor with the size of 20 x 7 x 5 x 5.

```

As you can see from the above, the number of channels changes from 3 to 7. The reasoning behind this change is that the convolution blocks will attempt to extract 7 different filters from the input channel.

# Conclusion

The pooling blocks and convolution blocks are important parts for the convolutional neural networks. These blocks allow you to extract useful features and could be used to reduce the size of the input tensor, potentially leading to faster training times.

That's all for today!
