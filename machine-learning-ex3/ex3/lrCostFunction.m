function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

h_theta = sigmoid(sum(theta' .* X, 2));
J = - (sum(y .* log(h_theta)) + sum((1-y) .* log(1 - h_theta))) / m;
cost_regularization = (lambda / (2 * m)) * sum(theta(2:end) .^ 2);
J = J + cost_regularization; % after implementing regularization
normal_grad = sum((h_theta - y) .* X) / m;

% T done in order to add with normal_grad
% Also, we don't require first element for regularization
% So we set the value of first element right away
grad(1) = normal_grad(1);
grad_regularization = (theta(2:end) .* (lambda / m))';
grad(2:end) = normal_grad(2:end) + grad_regularization;
grad(1) = normal_grad(1); % Since there is no impact of regularization on g0

% =============================================================

grad = grad(:);

end
