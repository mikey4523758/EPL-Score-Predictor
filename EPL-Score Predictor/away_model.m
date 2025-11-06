%% away_model.m
% Premier League Match Score Prediction - Away Team Model
%
% Author: Michael Lopez
% Description: Implements locally-weighted linear regression to predict 
%              away team scores using historical Premier League data.
%              Uses 20+ years of match statistics including goals for/against,
%              possession, and other performance metrics.
%

%% Load Training Data
% Read in historical away team performance data
D = readmatrix('away_model_data.csv');

y = D(:,11);  % Response variable: actual away team scores
X = D(:,1:10); % Feature matrix: 10 performance metrics per match

% Standardize features to have mean=0 and variance=1
% This helps with numerical stability and makes all features comparable
X = standardizedata(X);

n = size(X,1);  % Number of training samples
p = size(X,2);  % Number of features (10)

%% Load Test Data
% Read in the test set to evaluate model performance
D = readmatrix('away_target_data.csv');

z = D(:,11);  % True scores we're trying to predict
T = D(:,1:10); % Test features

% Apply same standardization to test data
T = standardizedata(T);

m = size(T,1);  % Number of test samples (110 games)

%% Prepare Design Matrices
% Add bias term (column of ones) for intercept in regression
Xdes = [X ones(n,1)];  % Training design matrix
Tdes = [T ones(m,1)];  % Test design matrix

%% Locally-Weighted Linear Regression (LWLR)
% For each test point, fit a local model weighted by similarity
% h=0.275 controls how "local" the regression is (kernel bandwidth)
% Smaller h = more local, larger h = more global
z_hat = zeros(m,1);  % Store predictions

for i = 1:m
    % Predict score for test game i using LWLR
    z_hat(i) = regress_val_local(y, Xdes, 0.275, Tdes(i,:)');
end

%% Evaluate Model Performance
% Scatter plot: actual vs predicted scores
scatter(z, z_hat)
axis([0 5 0 5])  % Most Premier League scores fall in 0-5 range

% Calculate sum of squared errors
err = sumsqerr(z, z_hat)  % Continuous predictions

% Round predictions to nearest integer (can't score 2.3 goals!)
z_hat_round = round(z_hat, 0);
err_round = sumsqerr(z, z_hat_round)  % Error with rounded scores

% Note: Model performs best on low-scoring games (0-2 goals)
% Higher-scoring outliers (4+ goals) are harder to predict accurately
%eof