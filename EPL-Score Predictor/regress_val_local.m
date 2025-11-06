function [z_hat b_hat] = regress_val_local(y,X,h,t)
% regress_val_local - Locally-Weighted Linear Regression (LWLR) prediction
%
% Author: Michael Lopez
% Description: Predicts a response value by fitting a local linear model
%              around the test point t. Unlike global regression which uses
%              all data equally, LWLR gives more weight to nearby training
%              examples. This is a non-parametric approach that adapts to
%              local patterns in the data.
%
% Inputs:
%    y - n-by-1 vector of training response values (actual scores)
%    X - n-by-p training design matrix (includes bias column)
%    h - Kernel bandwidth parameter (scalar, controls locality)
%        Small h (~0.1): very local, high variance, low bias
%        Large h (~1.0): more global, low variance, high bias
%        Our choice (0.275): balanced based on cross-validation
%    t - p-by-1 test point feature vector (point where we predict)
%
% Outputs:
%    z_hat - Predicted response value at test point t
%    b_hat - p-by-1 local regression coefficients used for this prediction
%            (Different for each test point - that's why it's "local"!)
%
% Algorithm:
%    1. Compute weight for each training point based on distance to t
%    2. Fit weighted linear regression using these weights
%    3. Evaluate fitted model at test point t
%
% Intuition:
%    Imagine predicting a team's score in an upcoming match. LWLR looks at
%    historically similar matches (similar possession, recent form, etc.)
%    and weights them higher. A match from 10 years ago with very different
%    stats gets little influence on the prediction.
%
% Complexity:
%    O(n*p + p^3) per prediction - expensive for large datasets!
%    Must solve a weighted regression for every test point.

%% Compute Weights for Each Training Example
n = size(y, 1);  % Number of training points
W = zeros(n);    % Initialize diagonal weight matrix

% For each training point, compute its similarity to test point t
for i = 1:n
    % W(i,i) is weight for training example i
    % Uses Gaussian kernel: closer points get weights near 1,
    % distant points get weights near 0
    W(i, i) = gaussiankernel(X(i, :)', t, h);
end

%% Fit Local Weighted Linear Regression
% Solve weighted least squares: min ||sqrt(W)*(y - Xb)||^2
% Training examples close to t (high W(i,i)) have more influence
b_hat = regress_fit_weighted(y, X, W);

%% Predict at Test Point
% Linear prediction: y = x'b (dot product of features and coefficients)
z_hat = b_hat' * t;

% Note: b_hat is specific to this test point t!
% For a different test point, we'd compute different weights and get
% different coefficients. That's the "local" part of LWLR.

return
%eof