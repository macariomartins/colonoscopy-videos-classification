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
