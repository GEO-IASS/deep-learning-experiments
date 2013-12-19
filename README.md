# Deep Learning Experiments

My solutions for Stanford's [UFLDL Tutorial](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial).

## Sparse Autoencoder

Back propagation algorithm applied to neural networks.

Full readings and file dependencies refer to [Exercise:Sparse Autoencoder](http://ufldl.stanford.edu/wiki/index.php/Exercise:Sparse_Autoencoder).

Note: This implementation of sparse autoencoder has already been vectorized, there is also an unvectorized implementation commented out in the code file.

## Preprocessing: PCA and Whitening

Full readings and file dependencies refer to [Exercise:PCA and Whitening](http://ufldl.stanford.edu/wiki/index.php/Exercise:PCA_and_Whitening).

## Softmax Regression

Use softmax regression to build a softmax classifier for handwritten digits classification or other classification problems.

Full readings and file dependencies refer to [Exercise:Softmax Regression](http://ufldl.stanford.edu/wiki/index.php/Exercise:Softmax_Regression).

## Self Taught Learning

Use sparse autoencoder to extract features from the original input data, these features are then fed into a softmax classifier for classification.

Full readings and file dependencies refer to [Exercise:Self-Taught Learning](http://ufldl.stanford.edu/wiki/index.php/Exercise:Self-Taught_Learning).

## Stacked Autoencoder

Building Deep Networks for Classification. Instead of using one level of sparse autoencoder for feature extraction, here we stacked two autoencoders to extract features, which are then fed into a softmax classifier. 
By adopting back-propagation fine-tuning in the deep network, one can achieve much better accuracy.

Full readings and file dependencies refer to [Exercise: Implement deep networks for digit classification](http://ufldl.stanford.edu/wiki/index.php/Exercise:_Implement_deep_networks_for_digit_classification).

## Linear Decoders with Autoencoders

Sparse autoencoder with linear activation (instead of sigmoid) function at its output layer.

Full readings and file dependencies refer to [Exercise:Learning color features with Sparse Autoencoders](http://ufldl.stanford.edu/wiki/index.php/Exercise:Learning_color_features_with_Sparse_Autoencoders).


&copy; 2013 Daogan Ao &lt;wvvwwwvvvv@gmail.com&gt;
