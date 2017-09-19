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

% Basic for loop implementation to find Delta_1 & Delta_2
%{
eyeMatrix = eye(num_labels);

% Calculate Cost and grad
% Steps mentioned for calculating grad
for row = 1:m
	% Step 1: Calculate a_1, z_2, a_2, z_3, a_output
	a_1 = [1; X(row,:)'];
	y_row = eyeMatrix(y(row),:)';

	% z_2 is vector of all sums in layer 3
	% a_2 is the hidden layer action vector
	z_2 = sum(bsxfun(@times, Theta1, a_1'), 2);
	a_2 = [1; sigmoid(z_2)];

	% z_3 is vector of all sums in layer 3
	% a_output is the k-dimensional vector that comes out of output layer
	z_3 = sum(bsxfun(@times, Theta2, a_2'), 2);
	a_output = sigmoid(z_3);

	% Cost function without regularization
	J = J - ((sum(bsxfun(@times, y_row, log(a_output))) ...
	+ sum(bsxfun(@times, (1-y_row), log(1 - a_output)))) / m);

	% Step 2: Calculate delta_3
	delta_3 = a_output - y_row;

	% Step 3: For hidden layer, calculate delta_2
	delta_2 = bsxfun(@times, (Theta2' * delta_3), sigmoidGradient([1; z_2]));

	% Step 4: Gather Delta_2 and Delta_1
	Theta2_grad = Theta2_grad + ((delta_3 * a_2') / m);
	Theta1_grad = Theta1_grad + ((delta_2(2:end) * a_1') / m);
end
%}

% Matrix implementation to find Delta_1 & Delta_2
eyeMatrix = eye(num_labels);
y_binary = zeros(size(y, 1), num_labels);

% Make a matrix to hold binary values of y
for label = 1:num_labels
	y_binary(y==label,:) = repmat(eyeMatrix(label,:), size(y_binary(y==label), 1), 1);
end

% Determine a_output for all training examples
% Below steps from 1 through 5 calculate theta grads
% Step 1: Calculate a_1, z_2, a_2, z_3, a_output
a_1 = [ones(m, 1), X]'; % n + 1 x m

% a_2 is the hidden layer action vector
z_2 = Theta1 * a_1; % hidden_layer_size x m
a_2 = [ones(1,m); sigmoid(z_2)]; % hidden_layer_size + 1 x m

% a_output is the k-dimensional vector that comes out of output layer
z_3 = Theta2 * a_2; % num_labels x m
a_output = sigmoid(z_3); % num_labels x m

% Step 2: Calculate delta_3
delta_3 = a_output - y_binary'; % num_labels x m

% Step 3: For hidden layer, calculate delta_2
% size = hidden_layer_size + 1 x m
delta_2 = bsxfun(@times, (Theta2' * delta_3), sigmoidGradient([ones(1,m); z_2]));

% Cost function without regularization
J = sum(-sum(bsxfun(@times, y_binary', log(a_output)) +
bsxfun(@times, (1-y_binary)', log(1-a_output))), 2) / m;

% Step 4: Gather Delta_2 and Delta_1 from all examples
for n = 1:m
	Theta2_grad = Theta2_grad + ((delta_3(:,n) * a_2(:,n)') / m);
	Theta1_grad = Theta1_grad + ((delta_2(2:end,n) * a_1(:,n)') / m);
end

% Cost regularization remains same for both matrix implementation
% and basic implementation
% Cost regularization function, J
cost_regularization = (lambda / (2 * m)) * ...
(sum(sum(bsxfun(@power, Theta1(:,2:end), 2), 2)) + ...
sum(sum(bsxfun(@power, Theta2(:,2:end), 2), 2)));
J = J + cost_regularization; % after implementing regularization

% Add regularization to all the terms other than first
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda / m) * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda / m) * Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
