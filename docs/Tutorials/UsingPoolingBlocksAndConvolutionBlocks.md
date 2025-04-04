# Using Pooling Blocks And Convolution Blocks

Pooling blocks and convolution blocks allow use to capture spatial information from the data. These blocks can be confusing to use at first, but in this tutorial, we will show you how those blocks work.

## Requirements

* An understanding on these two tutorials:

  * [General Tensor Conventions](GeneralTensorConventions.md)

  * [Spatial Dimension, Kernel And Stride](SpatialDimensionKernelAndStride.md)

## The Pooling Blocks

In order to construct our pooling block object, we must use the code shown below.

```lua

local AveragePooling2D = DataPredict.PoolingBlocks.AveragePooling2D.new({kernelDimensionSizeArray = {2, 2}, strideDimensionSizeArray = {2, 2})

```
