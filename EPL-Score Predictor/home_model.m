% home_model.m
% Premier League Match Score Prediction - Home Team Model
%
% Author: Michael Lopez
% Description: Implements locally weighted linear regression to predict 
%              home team scores using historical Premier League data.
%              Home teams typically have an advantage (home field advantage),
%              so we train separate models for home vs away predictions.
%

%% Load Training Data
% Read in historical home team performance data
D = readmatrix('home_model_data.csv');

y = D(:,11);  % Response variable: actual home team scores
X = D(:,1:10); % Feature matrix: 10 performance metrics per match

% Standardize features to have mean=0 and variance=1
% Critical for fair comparison across different scales (e.g., goals vs possession %)
X = standardizedata(X);

n = size(X,1);  % Number of training samples
p = size(X,2);  % Number of features (10)

%% Load Test Data
% Read in the test set to evaluate model performance
D = readmatrix('home_target_data.csv');

z = D(:,11);  % True scores we're trying to predict
T = D(:,1:10); % Test features

% Apply same standardization to test data (must use same scale as training!)
T = standardizedata(T);

m = size(T,1);  % Number of test samples (110 games)

%% Prepare Design Matrices
% Add bias term (column of ones) for intercept in regression
% This allows model to predict non-zero values even when features are zero
Xdes = [X ones(n,1)];  % Training design matrix
Tdes = [T ones(m,1)];  % Test design matrix

%% Locally-Weighted Linear Regression (LWLR)
% Non-parametric approach: fit a different model at each test point
% h=0.275 is the kernel bandwidth (tuned through cross-validation)
% This value balances bias-variance tradeoff for our dataset
z_hat = zeros(m,1);  % Store predictions

for i = 1:m
    % For each test game, fit a local regression model
    % Training games similar to test game i get higher weights
    z_hat(i) = regress_val_local(y, Xdes, 0.275, Tdes(i,:)');
end

%% Evaluate Model Performance
% Visual check: predictions should cluster around the y=x line
scatter(z, z_hat)
axis([0 5 0 5])  % Most Premier League scores fall in 0-5 range for visualization purposes

% Calculate sum of squared errors (SSE)
err = sumsqerr(z, z_hat)  % Continuous predictions

% Round predictions to nearest integer (actual soccer scores are integers)
z_hat_round = round(z_hat, 0);
err_round = sumsqerr(z, z_hat_round)  % Error with rounded scores

% Performance Notes:
% - Home teams tend to score slightly more than away teams
% - Model achieves ~1 goal average deviation across 110 test games
% - Works best for typical scores (1-3 goals), struggles with outliers (4+)
%eof