# Getting Started

In this library, we can customize many of our models, optimizers and others to fit our needs. This was made possible thanks to the object-orientated design of our library.

To start, we must first link our deep learning library with our tensor library. However, you must use "Aqwam's Tensor Library" as every calculations made by our models are based on that tensor library.

| Version                                            | Beta Version                                                                                                                                                    |
|----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Deep Learning Library (DataPredict Neural)         | [Aqwam's Deep Learning Library](https://github.com/AqwamCreates/DataPredict-Neural/blob/main/module_scripts/AqwamDeepLearningLibrary.rbxm)                      |
| Tensor Library (TensorL)                           | [Aqwam's Tensor Library](https://github.com/AqwamCreates/TensorL/blob/main/TensorL_Table.lua)                                                                   |
| Tensor Library - Efficient (TensorL Efficient)     | [Aqwam's Tensor Library](https://github.com/AqwamCreates/TensorL/blob/main/TensorL_Table.lua)                                                                   |

To download the files from GitHub, you must click on the download button highlighted in the red box.

![Github File Download Screenshot](https://github.com/AqwamCreates/DataPredict/assets/67371914/b921d568-81b9-4f47-8a96-e0ab0316a4fe)

Then drag the files into Roblox Studio from a file explorer of your choice.

Once you put those two libraries into your game, make sure you link the Deep Learning Library with the Tensor Library. This can be done via setting the “AqwamTensorLibraryLinker” value (under the Deep Learning library) to the Tensor Library.

![Screenshot 2024-06-08 071322](https://github.com/AqwamCreates/DataPredict-Neural/assets/67371914/c4ccb9b9-4c02-4708-bffd-5959e73d99f0)

Next, we will use require() function to our deep learning library:

```lua

local DataPredictNeural = require(AqwamDeepLearningLibrary) 

```
