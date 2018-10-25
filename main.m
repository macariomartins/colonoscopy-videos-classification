% Colonoscopy Lesions Classification
% Author: Macário Martins <macariomartinsjunior@gmail.com>

%% Cleaning the workspace
%
% Just a few commands to close previous windows, clear the workspace and
% clear the command prompt.
%
close all;
clear;
clc;

%% Loading database and Settings
%
%  Bellow you wil find configurations to be used in neural networks. Feel
%  free to test any configuration you want.
%
load('Dataset\gastrointestinal_colonoscopy_lesions_dataset.mat');

all_lights  = find(light_type ~= 0);
white_light = find(light_type == 1);
nbi_light   = find(light_type == 2);

X             = features(:, nbi_light); % P-by-N matrix of P features and N input vectors
D             = class_label(nbi_light); % R-by-N matrix of R labels and N target class vectors
err           = 1e-7;                    % Mean squared error goal for both neural networks
epochs        = 500;                     % Max number of epochs
learning_rate = 0.01;                    % Learning rate to be used in weights adjusts

%% Build Neural Networks
%
%  The functions bellow build the neural networks with global parameters,
%  i.e, parameters that could be used in both neural networks. The
%  parameters in brackets are the number of neurons in hidden layers.
%
mlp = NeuralNetworks.MLP(X, D, [7, 5], learning_rate, epochs, err);
rbf = NeuralNetworks.RBF(X, D, [5], learning_rate, epochs, err);

%% Validations
%
%  The lines bellow makes the cross-validation with Leave-One-Out (LOO) and
%  K-Fold with K = 10 (default). You may set a number of trials as well, to
%  search for the best and worst cases for k-fold validation. Be carefull
%  when selecting the number of trials. Higher numbers will need several
%  time to be finished.
%
k      = 10; % Use it for k-fold
trials = 10; % Number of times the validations will be called

% Variables bellow store the accuracies for each trial in both LOO and
% KFold validations.
mlp_accuracies = zeros(2, trials);
rbf_accuracies = zeros(2, trials);

% The following variabales store the confusion matrices for each LOO and
% KFold validations. Each confusion matrix is already a mean matrix of the
% validation internal executions.
mlp_confusions = zeros(max(D), max(D), trials);
rbf_confusions = zeros(max(D), max(D), 2, trials);

% "loo_ind" and "kfold_ind" are just names to the indexes. It is good for
% better understanding of the attributions returned from validation methods
loo_ind   = 1;
kfold_ind = 2;

for t = 1:trials
    fprintf("Trial %2d/%2d", t, trials);
    fprintf("\n--------------------------");
    
%     fprintf("\n\tMLP - LOO: ");
%     [accuracy, confusion]            = Validations.LOO(mlp, X, D);
%     mlp_accuracies(loo_ind, t)       = accuracy;
%     mlp_confusions(:, :, loo_ind, t) = confusion;
%     fprintf("%.4f", mlp_accuracies(loo_ind, t));
%     
%     fprintf("\n\tRBF - LOO: ");
%     [accuracy, confusion]            = Validations.LOO(rbf, X, D);
%     rbf_accuracies(loo_ind, t)       = accuracy;
%     rbf_confusions(:, :, loo_ind, t) = confusion;
%     fprintf("%.4f", rbf_accuracies(loo_ind, t));
%     
%     fprintf("\n\tMLP - %d-Fold: ", k);
%     [accuracy, confusion]              = Validations.KFold(mlp, X, D, k);
%     mlp_accuracies(kfold_ind, t)       = accuracy;
%     mlp_confusions(:, :, kfold_ind, t) = confusion;
%     fprintf("%.4f", mlp_accuracies(kfold_ind, t));
    
    fprintf("\n\tRBF - %d-Fold: ", k);
    [accuracy, confusion]              = Validations.KFold(rbf, X, D, k);
    rbf_accuracies(kfold_ind, t)       = accuracy;
    rbf_confusions(:, :, kfold_ind, t) = confusion;
    fprintf("%.4f\n\n", rbf_accuracies(kfold_ind, t));
end

%% Results
%
%  The lines bellow collect the results for MLP and RBF classifications,
%  showing the best, worst cases and the mean confusion matrix for each
%  validation method.
%
mlp_best_cases     = max(mlp_accuracies, [], 2);
rbf_best_cases     = max(rbf_accuracies, [], 2);
mlp_worst_cases    = min(mlp_accuracies, [], 2);
rbf_worst_cases    = min(rbf_accuracies, [], 2);
mlp_mean_confusion = mean(mlp_confusions, 4);
rbf_mean_confusion = mean(rbf_confusions, 4);

