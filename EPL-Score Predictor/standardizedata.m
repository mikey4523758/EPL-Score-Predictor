function [Xs V S Xc] = standardizedata(X)
% standardizedata - Z-score normalization of features
%
% Author: Michael Lopez
% Description: Transforms each feature to have mean=0 and variance=1.
%              Critical preprocessing step that puts all features on the
%              same scale, preventing features with large ranges from
%              dominating the model.
%
% Inputs:
%    X - n-by-p data matrix (n samples, p features)
%
% Outputs:
%    Xs - n-by-p standardized data matrix (mean=0, variance=1 per column)
%    V  - p-by-p diagonal variance matrix (original variances on diagonal)
%    S  - p-by-p standardizing matrix (scaling operator)
%    Xc - n-by-p centered data matrix (mean=0, but original variance)
%
% Standardization:
%    After standardization, both features have comparable influence.

%% Get Dimensions
n = size(X, 1);  % Number of samples (games)
p = size(X, 2);  % Number of features (stats)

%% Step 1: Center the Data (subtract mean from each column)
% This shifts each feature to have mean = 0
Xc = centerdata(X);

%% Step 2: Compute Variance Matrix
% Variance measures spread of each feature
V = zeros(p);  % Initialize p-by-p matrix

for i = 1 : p
    Xci = Xc(:, i);  % i-th centered feature column
    
    % Sample variance formula: Var(X) = sum((X - mean(X))^2) / (n-1)
    % Since Xc is already centered, this simplifies to: Xci'*Xci / (n-1)
    Vii = ((Xci.')*Xci)/(n-1);
    V(i, i) = Vii;  % Store variance on diagonal
end

%% Step 3: Construct Standardizing Matrix
% S scales each feature by 1/std = 1/sqrt(variance)
% sqrtm(V) gives matrix with sqrt(variance) on diagonal
% inv(sqrtm(V)) gives matrix with 1/sqrt(variance) on diagonal
S = inv(sqrtm(V));

%% Step 4: Apply Standardization
% Xs = Xc * S scales each centered column by its standard deviation
% Each column of Xs now has mean=0 and variance=1
Xs = Xc * S;

% Verification: mean(Xs) should be ~[0, 0, ..., 0]
%               var(Xs) should be ~[1, 1, ..., 1]

return
%eof