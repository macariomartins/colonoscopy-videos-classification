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

% GLOBAL SETTGINGS
X   = features;     % P-by-N matrix of P features and N input vectors
D   = class_label;  % R-by-N matrix of R labels and N target class vectors
err = 1e-7;         % Mean squared error goal for both neural networks

% MLP SETTINGS
Q = [7, 5]; % Row vector of one or more hidden layer sizes.

% RBF SETTINGS
spread = 1;           % Spread of radial basis functions (default = 1.0)
max_q  = size(X, 2);  % Maximum number of neurons (default is N)
increment_q = 1;      % Number of neurons to add between displays (default = 1)

%% Build Neural Networks
% mlp = NeuralNetworks.MLP(X, D, Q, err);
% rbf = NeuralNetworks.RBF(X, D, err, spread, max_q, increment_q);
