function b_hat = regress_fit(y,X)
% regress_fit - Ordinary Least Squares (OLS) linear regression
%
% Author: Michael Lopez
% Description: Finds the best-fit linear relationship between features X
%              and response y by minimizing sum of squared errors.
%              This is the classic regression approach.
%
% Inputs:
%    y - n-by-1 vector of response values (e.g., actual match scores)
%    X - n-by-p design matrix (feature matrix, usually includes bias column)
%        n = number of training examples
%        p = number of features (including intercept if using X = [features, ones])
%
% Outputs:
%    b_hat - p-by-1 vector of regression coefficients
%            These are the optimal weights for linear combination:
%            y_predicted = X * b_hat
%
% Assumptions:
%    - Linear relationship between X and y
%    - X'X is invertible (features are linearly independent)
%    - Errors are independent and identically distributed
%
% Note: For our project, this is used within LWLR, where we fit
%       many local OLS models rather than one global model.

%% Solve Normal Equations
% (X'X)^-1 * X'y gives least squares solution
% This is the closed-form solution - no iteration needed!
b_hat = inv(X' * X) * X' * y;

% Alternative: b_hat = X \ y (MATLAB's built-in, more numerically stable)
% We use explicit form here for educational clarity

return
%eof