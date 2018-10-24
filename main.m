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

X             = features;    % P-by-N matrix of P features and N input vectors
D             = class_label; % R-by-N matrix of R labels and N target class vectors
err           = 1e-7;        % Mean squared error goal for both neural networks
epochs        = 500;         % Max number of epochs
learning_rate = 0.01;        % Learning rate to be used in weights adjusts

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
%  search for the best and worse cases for k-fold validation. Be carefull
%  when selecting the number of trials. Higher numbers will need several
%  time to be finished.
%
k      = 10;  % Use it for k-fold
trials = 100; % Number of times the validations will be called

mlp_accuracies = zeros(trials, 2);
rbf_accuracies = zeros(trials, 2);

for i = 1:trials
    fprintf("Trial %3d/%3d", i, trials);
    fprintf("\n-------------");
    
    fprintf("\n\tMLP - LOO: ");
    mlp_accuracies(i, 1) = Validations.LOO(mlp, X, D);
    fprintf("%.4f", mlp_accuracies(i, 1));
    
    fprintf("\n\tMLP - %d-Fold: ", k);
    mlp_accuracies(i, 2) = Validations.KFold(mlp, X, D, k);
    fprintf("%.4f", mlp_accuracies(i, 2));
    
    fprintf("\n\tRBF - LOO: ");
    rbf_accuracies(i, 1) = Validations.LOO(rbf, X, D);
    fprintf("%.4f", rbf_accuracies(i, 1));
    
    fprintf("\n\tRBF - %d-Fold: ", k);
    rbf_accuracies(i, 2) = Validations.KFold(rbf, X, D, k);
    fprintf("%.4f\n\n", rbf_accuracies(i, 2));
end

mlp_worse_case = min(mlp_accuracies(:, 1));
mlp_best_case  = max(mlp_accuracies(:, 1));

rbf_worse_case = min(rbf_accuracies(:, 1));
rbf_best_case  = max(rbf_accuracies(:, 1));

