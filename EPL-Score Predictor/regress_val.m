function y_hat = regress_val(b_hat,X)
% regress_val - Make predictions using fitted regression model
%
% Author: Michael Lopez
% Description: Evaluates a trained linear regression model on new data.
%              Given coefficients b_hat and feature matrix X, computes
%              predictions y_hat = X * b_hat. Simple but essential function!
%
% Inputs:
%    b_hat - p-by-1 vector of regression coefficients
%            Learned from training data via regress_fit or other method
%            Last element is typically the intercept (bias term)
%    X     - n-by-p design matrix (test/validation data)
%            n = number of examples to predict
%            p = number of features (must match b_hat length)
%            Should include ones column if model was trained with intercept
%
% Outputs:
%    y_hat - n-by-1 vector of predicted response values
%            y_hat(i) = predicted score for example i
%
% Use Cases:
%    - Evaluate model on test set: compare y_hat to actual y
%    - Make predictions on new match data (forecasting)
%    - Cross-validation: predict on held-out folds
%
% Note: This is for standard (global) linear regression.
%       For LWLR, we use regress_val_local instead, which computes
%       different coefficients for each test point.

%% Compute Predictions
% Matrix multiplication: each row of X (one example) times b_hat (coefficients)
% gives one prediction. Vectorized for efficiency - no loops needed!
y_hat = X * b_hat;

% Example: If X is 110x11 (110 games, 10 features + bias) and
%          b_hat is 11x1, then y_hat will be 110x1 (one score per game)

return
%eof