# fB-DNN: Simultaneous Classification and Feature Selection for Complex Functional Data via Deep Neural Networks
------------------------------------------------

# Functional principal components
- For the j-th multi-dimensional functional covariates ![X](https://latex.codecogs.com/svg.image?X_j(s_1,\ldots,s_{d_j})), use Fourier basis functions to extract projection scores ![xi](https://latex.codecogs.com/svg.image?\xi_{j1},\xi_{j2},\ldots) by integration.
- Truncate the functional principal components ![truncation](https://latex.codecogs.com/svg.image?\xi_{j1},\ldots,\xi_{jr_j}).
------------------------------------------------

# Simultaneous classification and feature selection via High-dimensional BIC
------------------------------------------------

# Deep neural network hyperparameters and structures
## hyperparameters
- truncation of functional principal components
- number of layers 
- neurons per layer (uniform for all layers)
- dropout rate (data dependent)
- tuning for LASSO penalty
## other details for neural networks 
- Loss function: cross-entropy loss
- Batch size: data dependent
- Epoch number: data dependent
- Activation function: ReLU, with Sigmoid for the last layer
- Optimizer: Adam
- Regularizer: LASSO
-------------------------------------------------------------

# Function descriptions
-------------------------------------------------------------

# Examples
-------------------------------------------------------------
- "datagen_2d": simulated data for functional covariates in two-dimension.
- "datagen_1d2d" simulated data for functional covariates in both one-dimension and two-dimension.
