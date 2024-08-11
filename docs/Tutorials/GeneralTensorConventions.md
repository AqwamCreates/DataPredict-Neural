# General Tensor Conventions

When handling tensors, there are so many ways we can manipulate them. Because of this, DataPredict Neural have specific guidelines on what each values means in a tensor. I will talk about them below.

## The Basics Of Dimension Size Arrays 

You may have noticed that some of the function blocks requires you to input "dimensionSizeArray". These typically tell the number of dimensions and the sizes of each dimension. For example:

```lua

local dimensionSizeArray1 = {10, 3} -- Dimension 1 has the size of 10, dimension 2 has the size of 5.

local dimensionSizeArray2 = {7, 6, 9, 5} -- Dimension 1 has the size of 7, dimension 2 has the size of 6, dimension 3 has the size of 9 and dimension 4 has the size of 5.

```

## The Dimension Size Arrays Have Meanings!

Different function blocks that requires "dimensionSizeArray" requires different number of dimensions for the input tensors. In general:

| Dimension | Meaning                                   |
|-----------|-------------------------------------------|
| 1         | Number of data                            |
| 2         | Number of channels                        |
| N + 2     | Number of width, height, length and so on |

If you wish to add number of time steps, then they needed to be after the number of channels. If the number of channels does not exist, then they needed to be after the number of data. In general:

| Dimension | Meaning                                   |
|-----------|-------------------------------------------|
| 1         | Number of data                            |
| 2         | Number of channels                        |
| 3         | Number of time steps                      |
| N + 3     | Number of width, height, length and so on |

If you only want to use 2D tensors, then the rules will be slightly different. In general:

| Dimension | Meaning            |
|-----------|--------------------|
| 1         | Number of data     |
| 2         | Number of features |
	
That's pretty much it! I'll show you an example on how these tensors are interpreted below for the first and third rules.

```lua

local dimensionSizeArray1 = {20, 3, 4} -- There is 20 data, where each of them have three channels and a length of 4.

local dimensionSizeArray2 = {900, 10} -- There is 900 data, where each of them have 10 features.

```

Now you have managed to read all the general tensor conventions for the DataPredict Neural library.
