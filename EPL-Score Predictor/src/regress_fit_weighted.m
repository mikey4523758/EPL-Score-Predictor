function b_hat = regress_fit_weighted(y,X,W)
% regress_fit_weighted - Weighted least squares regression
%
% Author: Michael Lopez
% Description: Fits a linear regression model where each training example
%              has a different importance weight. Core of locally-weighted
%              regression (LWLR) where nearby points get more weight.
%
% Inputs:
%    y - n-by-1 vector of response values (actual scores)
%    X - n-by-p design matrix (feature matrix with bias column)
%        n = number of training examples
%        p = number of features (including bias term)
%    W - n-by-n diagonal weighting matrix
%        W(i,i) = importance of training example i
%        Typically from Gaussian kernel: closer examples → higher weights
%
% Outputs:
%    b_hat - p-by-1 vector of regression coefficients
%            [feature_1_coef, feature_2_coef, ..., intercept]'
%
% Use Case in LWLR:
%    When predicting at test point t, we compute weights W where:
%    W(i,i) = gaussiankernel(training_point_i, t, h)
%    This makes the regression "local" to the test point.

%% Solve Weighted Least Squares
% pinv() is pseudoinverse - more stable than inv() for ill-conditioned matrices
% This can happen when some weights are near zero (far away training points)
b_hat = pinv(X' * W * X) * X' * W * y;

% Note: The multiplication order matters for efficiency
% (X'W)X is faster than X'(WX) when n >> p

return
%eof