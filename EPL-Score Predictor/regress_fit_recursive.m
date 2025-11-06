function [b_hat P p] = regress_fit_recursive(y,x,P,p,lambda)
% regress_fit_recursive - Online learning via Recursive Least Squares
%
% Author: Michael Lopez
% Description: Updates regression coefficients incrementally as new data
%              arrives. Useful for real-time predictions or when data doesn't
%              fit in memory. Uses Sherman-Morrison formula for efficient
%              matrix inversion updates.
%
% Inputs:
%    y      - Scalar response value for current data point
%    x      - p-by-1 feature vector for current data point
%    P      - p-by-p inverse covariance matrix from previous data
%             (Initialize: P = large_value * I for first point)
%    p      - p-by-1 covariance vector from previous data
%             (Initialize: p = zeros(p,1) for first point)
%    lambda - Forgetting factor (0 < lambda <= 1)
%             lambda = 1: all data weighted equally (standard RLS)
%             lambda < 1: recent data weighted more (adaptive to change)
%
% Outputs:
%    b_hat - p-by-1 updated regression coefficients
%    P     - p-by-p updated inverse covariance matrix
%    p     - p-by-1 updated covariance vector
%
% Algorithm:
%    Instead of recomputing (X'X)^-1 * X'y from scratch each time,
%    we update P and p incrementally. This is O(p^2) instead of O(np^2).
%
% Use Case:
%    When match data streams in game-by-game, we can update predictions
%    without retraining on all historical data.

%% Update Covariance Vector
% Accumulate new observation's contribution to X'y
% Lambda < 1 gradually "forgets" old data by exponential decay
p = lambda * p + x * y;

%% Update Inverse Covariance Matrix
% Sherman-Morrison formula for efficient rank-1 update of matrix inverse
% This avoids expensive full matrix inversion at each step
P = (1/lambda) * (P - (P * x * x' * P)/(lambda + x' * P * x));

%% Compute Updated Regression Coefficients
% b_hat = (X'X)^-1 * X'y, but computed incrementally using P and p
% This gives us the same answer as batch least squares, but faster!
b_hat = P * p;

return
%eof