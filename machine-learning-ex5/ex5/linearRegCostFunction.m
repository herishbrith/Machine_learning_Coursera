function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.

error_term = (theta' * X')' - y;
J = sum(bsxfun(@power, error_term, 2)) / (2 * m);
J_reg = sum(bsxfun(@power, theta(2:end), 2)) * lambda / (2 * m);
J = J + J_reg;

grad = (error_term' * X)' / m;
grad_reg = theta(2:end) * lambda / m;
grad(2:end) = grad(2:end) + grad_reg;

% =========================================================================

grad = grad(:);

end
