function w = gaussiankernel(x1, x2, h)
% gaussiankernel - Computes similarity between two data points
%
% Author: Michael Lopez
% Description: Gaussian kernel function for measuring similarity between
%              data points in locally-weighted regression. Returns a weight
%              between 0 and 1, where 1 means identical points and values
%              closer to 0 mean very different points.
%
% Inputs:
%    x1 - First data vector (p-by-1)
%    x2 - Second data vector (p-by-1)
%    h  - Kernel bandwidth/locality parameter (scalar)
%         Controls how quickly similarity drops with distance:
%         - Small h: only very nearby points get high weights (more local)
%         - Large h: points further away still get significant weights (more global)
%
% Outputs:
%    w - Kernel similarity score (scalar between 0 and 1)
%
% Math:
%    w = exp(-0.5 * (x1-x2)' * H * (x1-x2))
%    where H = h*I is the bandwidth matrix
%    This is essentially a multivariate Gaussian centered at x2
%
% Example:
%    If x1 = x2, then w = 1 (maximum similarity)
%    As ||x1 - x2|| increases, w approaches 0 (less similar)

%% Construct Bandwidth Matrix
% H controls the "reach" of the kernel in each dimension
% Using h*I means same bandwidth in all feature directions (isotropic)
H = h.*eye(length(x1));

%% Compute Gaussian Kernel Weight
% This is the "bell curve" similarity measure
% The exponent measures squared Mahalanobis distance scaled by H
% exp(-0.5*distance^2) gives us a smooth, differentiable weight function
w = exp(-0.5*(x1 - x2)'*H*(x1 - x2));

% Note: In LWLR, this weight determines how much influence a training point
% has when predicting at a test point. Closer training examples get more weight.

return
%eof