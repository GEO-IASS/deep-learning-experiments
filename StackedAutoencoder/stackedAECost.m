function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
numCases = size(data, 2);
groundTruth = full(sparse(labels, 1:numCases, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

depth = numel(stack);
activations = cell(depth+1, 1);
delta = cell(depth+1, 1);
activations{1} = data;

% compute the activations of each leayer in the network
for i = 1:depth, 
    weightSum = stack{i}.w * activations{i} + ... 
        repmat(stack{i}.b, [1, size(activations{i}, 2)]);
    activations{i+1} = sigmoid(weightSum);
end

M = softmaxTheta * activations{depth+1};
% optimization to prevent overflow
M = bsxfun(@minus, M, max(M, [], 1));
% M[r][c] is exp(theta(r)'* x(c))
M = exp(M);
% P[numClasses * m] : probability matrix
% P[c][i] is the probabilities of data(:,i) == class(c)
P = bsxfun(@rdivide, M, sum(M));

cost = -(1/numCases) * groundTruth(:)' * log(P(:)) + ... 
    lambda/2 * sum(softmaxTheta(:) .^ 2);

softmaxThetaGrad = -(1/numCases) * ((groundTruth - P) * ... 
    activations{depth+1}') + lambda*softmaxTheta;

% delta of softmax layer's output
delta{depth+1} = -(softmaxTheta' * (groundTruth - P)) .* ...
    (activations{depth+1} .* (1-activations{depth+1}));

% compute the delta of each activation layer
for i = depth:-1:2, 
    delta{i} = stack{i}.w' * delta{i+1} .* (activations{i} .* (1-activations{i}));
end

% compute the gradient of each weight matrices in the network
for i = depth:-1:1, 
    stackgrad{i}.w = (1/numCases) * delta{i+1} * activations{i}';
    stackgrad{i}.b = (1/numCases) * sum(delta{i+1}, 2);
end


% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
