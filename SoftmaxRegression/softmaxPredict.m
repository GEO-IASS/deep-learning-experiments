function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

% M[numCases * m], M[r][c] is theta(r)'* x(c)
% M = theta * data;
% 
% % optimization to prevent overflow
% M = bsxfun(@minus, M, max(M, [], 1));
% % M[r][c] is exp(theta(r)'* x(c))
% M = exp(M);
% % hypothysis matrix 
% % now P[i][j] is the propability of x(j) == class(i)
% P = bsxfun(@rdivide, M, sum(M));
% [maxVal, pred] = max(P);

[maxVal, maxValIndex] = max(theta * data);
pred = maxValIndex;


% ---------------------------------------------------------------------

end

