function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m, 1) X];
eyeMatrix = eye(num_labels);

% Calculate input to hidden layer neurons, a_hidden
for row = 1:m
	x_row = X(row,:);
	y_row = eyeMatrix(y(row),:);

	% a_hidden is the hidden layer action vector
	a_hidden = [1; sigmoid(sum(bsxfun(@times, Theta1, x_row), 2))];

	% a_output is the k-dimensional vector that comes out of output layer
	a_output = sigmoid(sum(bsxfun(@times, Theta2, a_hidden'), 2));

	% Cost function without regularization
	J = J - ((sum(bsxfun(@times, y_row', log(a_output))) ...
	+ sum(bsxfun(@times, (1-y_row)', log(1 - a_output)))) / m);

	% T done in order to add with normal_grad
	% Also, we don't require first element for regularization
	% Since there is no impact of regularization on g0
	% So we set the value of first element right away
	%normal_grad = sum(bsxfun(@times, (a_output - y_row), X)) / m;
	%grad(1) = normal_grad(1);
	%grad_regularization = (bsxfun(@times, theta(2:end), (lambda / m)))';
	%grad(2:end) = normal_grad(2:end) + grad_regularization;
end

% Cost regularization function for J
cost_regularization = (lambda / (2 * m)) * ...
(sum(sum(bsxfun(@power, Theta1(:,2:end), 2), 2)) + ...
sum(sum(bsxfun(@power, Theta2(:,2:end), 2), 2)));
J = J + cost_regularization; % after implementing regularization















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
