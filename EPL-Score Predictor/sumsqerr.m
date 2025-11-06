function e = sumsqerr(y,y_hat)
% sumsqerr - Calculate Sum of Squared Errors (SSE)
%
% Author: Michael Lopez
% Description: Computes the total squared prediction error. SSE measures how
%              well predictions match actual values.
%
% Inputs:
%    y     - n-by-1 vector of actual response values (ground truth)
%    y_hat - n-by-1 vector of predicted response values (model output)
%
% Outputs:
%    e - Scalar SSE value (always non-negative)
%        e = 0 means perfect predictions
%        Larger e means worse predictions
%
% Math:
%    SSE = sum((y - y_hat)^2) = (y - y_hat)' * (y - y_hat)
%    
%    Breaking it down:
%    - (y - y_hat) gives vector of residuals (errors)
%    - Squaring penalizes large errors more than small ones
%    - Sum gives total error across all predictions
%
% Related Metrics:
%    - MSE (Mean Squared Error) = SSE / n
%    - RMSE (Root Mean Squared Error) = sqrt(MSE)
%      RMSE is in same units as y (e.g., goals)
%    - R^2 = 1 - SSE/SST (proportion of variance explained)
%
% Square Errors Advantages:
%    1. Positive and negative errors don't cancel out
%    2. Larger errors are penalized more heavily quadratic penalty
%    3. Mathematically convenient ifferentiable, convex
%
% Interpretation Our Project:
%    If SSE = 110 over 110 games, then MSE = 1, RMSE = 1 goal
%    This means on average, predictions are off by ~1 goal per game
%    For Premier League typical scores 1-3, this is pretty good!
%

%% Compute Sum of Squared Errors
% Vectorized computation: (y-y_hat) is n-by-1 residual vector
% Transpose times itself gives sum of squared elements
e = (y-y_hat)'*(y-y_hat);

% Equivalent to: sum((y - y_hat).^2)
% But matrix form is more compact and efficient

return
%eof