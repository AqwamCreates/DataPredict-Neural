# Roadmap

## Core

The list of items shown below are likely to be implemented due to their mainstream use, ability to increase learning speed, or ability to reduce computational resources.

* None

## Nice-To-Have

The list of items shown below may not necessarily be implemented in the future. However, they could be prioritized with external demand, collaboration, or funding.

* Dilated Convolution Layers And Pooling Layers

  * Enables larger receptive field without more weight parameters.

  * Good in sparse-data settings.

  * Unknown use cases related to game environments.

* Generalized N-Dimensional Convolution Layers And Pooling Layers

  * Currently, we have up to 3 dimensional kernels.

  * Useful for pushing the boundaries of convolutional neural networks.

  * 4 dimensional kernels are used in videos. Unknown use cases for game environments.

* Less Bloated Function Blocks Design

  * Currently, differentiate() function have excessive amount of code being used. Additionally, we have suspicions that our initial code design decision might not have led to efficient backward propagation calculation.

  * However, the current design still enables model parallelism and data parallelism. As such, we are debating or not if there are tradeoff on code design with parellism flexibility.
