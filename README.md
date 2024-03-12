# fB-DNN: Simultaneous Classification and Feature Selection for Complex Functional Data via Deep Neural Networks
------------------------------------------------

# Functional principal components
- For the j-th multi-dimensional functional covariates ![X](https://latex.codecogs.com/svg.image?X_j(s_1,\ldots,s_{d_j})), use Fourier basis functions to extract projection scores ![xi](https://latex.codecogs.com/svg.image?\widehat{\xi}_{j1},\widehat{\xi}_{j2},\ldots) by integration.
- Truncate the functional principal components ![truncation](https://latex.codecogs.com/svg.image?\widehat{\xi}_{j1},\ldots,\widehat{\xi}_{jr_j}).
------------------------------------------------

# Simultaneous classification and feature selection via High-dimensional BIC
------------------------------------------------
- deep neural network class: ![X](https://latex.codecogs.com/svg.image?\mathcal{D}=\left\\{f_{b,W}:f=\sigma^\ast\left(\sum_{j=1}^p&space;b_j^\intercal\widehat{\xi}_j^{(r_j)}&plus;g_{W}\left(\widehat{\xi}_j^{(r_1)},\ldots,\widehat{\xi}_j^{(r_p)}\right)\right)\right\\})
- objective function: ![X](https://latex.codecogs.com/svg.image?\min_{f\in\mathcal{D}}n^{-1}\sum_{i=1}^n\mathcal{L}\left(Y_i,f\left(\widehat{\xi}_{i1}^{(r_1)},\ldots,\widehat{\xi}_{ip}^{(r_p)}\right)\right)&plus;\lambda\sum_{j=1}^p&space;P_j\left(b_j\right),\text{s.t.}\|W_1^{(j)}\|_\infty\leq&space;C\|b_j\|)

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
- "score_extract.py": using Fourier basis to obtain the functional principal components.
- "main.py": main functions for fB-DNN with cross-entropy loss.  

# Examples
-------------------------------------------------------------
- "datagen_2d.py": simulated data for functional covariates in two-dimension.
- "datagen_1d2d.py" simulated data for functional covariates in both one-dimension and two-dimension.
