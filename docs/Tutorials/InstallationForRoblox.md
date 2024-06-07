# Getting Started

In this library, we can customize many of our models, optimizers and others to fit our needs. This was made possible thanks to the object-orientated design of our library.

To start, we must first link our deep learning library with our tensor library. However, you must use "Aqwam's Tensor Library" as every calculations made by our models are based on that tensor library.

| Version                                        | Beta Version                                                                                                                                                    |
|------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Deep Learning Library (DataPredict Neural)     | [Aqwam's Deep Learning Library](https://github.com/AqwamCreates/DataPredict-Neural/blob/main/module_scripts/AqwamDeepLearningLibrary.rbxm)                      |
| 3D Tensor Library (TensorL3D)                  | [Aqwam's 3D Tensor Library](https://github.com/AqwamCreates/TensorL3D/blob/main/module_scripts/Aqwam3DTensorLibrary.rbxm)                                       |

To download the files from GitHub, you must click on the download button highlighted in the red box.

![Github File Download Screenshot](https://github.com/AqwamCreates/DataPredict/assets/67371914/b921d568-81b9-4f47-8a96-e0ab0316a4fe)

Then drag the files into Roblox Studio from a file explorer of your choice.

Once you put those two libraries into your game, make sure you link the Machine Learning Library with the Matrix Library. This can be done via setting the “AqwamMatrixLibraryLinker” value (under the Machine Learning library) to the Matrix Library.

![Screenshot 2023-12-11 011824](https://github.com/AqwamCreates/DataPredict/assets/67371914/f8dee5ef-edb0-455f-bf4a-5160ccbc35ef)

Next, we will use require() function to our deep learning library:

```lua

local DataPredictNeural = require(AqwamDeepLearningLibrary) 

```
