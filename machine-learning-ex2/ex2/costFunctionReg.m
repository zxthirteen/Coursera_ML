function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

htheta = sigmoid(X*theta);
n = length(theta);
J = sum(-y.*log(htheta) - (-y.+1).*log(-htheta.+1))/m + lambda/2/m*sum(theta([2, n]).^2);
diff = htheta-y;
tail = lambda/m*theta;
tail(1,1) = 0;
grad = (X'*diff)/m+tail;

% =============================================================

end
