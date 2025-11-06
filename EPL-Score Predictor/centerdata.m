function [Xc C] = centerdata(X)
% centerdata - Centers data by subtracting mean from each feature
%
% Author: Michael Lopez
% Description: Preprocessing step that transforms data to have zero mean.
%              This is crucial for many ML algorithms and helps with
%              numerical stability in regression.
%
% Inputs:
%    X - n-by-p data matrix (n samples, p features)
%
% Outputs:
%    Xc - n-by-p matrix of centered data (each column now has mean=0)
%    C  - n-by-n centering matrix (mathematical operator for centering)
%
% Example:
%    If X = [1 4; 2 5; 3 6], then each column is centered around its mean
%    Column 1 mean = 2, Column 2 mean = 5
%    Xc = [-1 -1; 0 0; 1 1]
%

%% Construct Centering Matrix
% The centering matrix C = I - (1/n)*ones(n,n) is a linear operator
% that subtracts the mean from each column when applied to X
n = size(X, 1);  % Number of data points
I = eye(n);      % n-by-n identity matrix
one = ones(n);   % n-by-n matrix of ones

% Mathematical centering operator: removes mean from each feature
C = I - (1/n)*one;

%% Apply Centering Transformation
% Xc = C * X centers each column (feature) by subtracting its mean
% This is equivalent to: Xc(:,i) = X(:,i) - mean(X(:,i)) for each column i
Xc = C * X;

return
%eof