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
%  Bellow you will find configurations to be used in neural networks. Feel
%  free to test any configuration you want.
%
load('Dataset\gastrointestinal_colonoscopy_lesions_dataset.mat');

%--------------------------------------------------------------------------
% Ilumination settings
%--------------------------------------------------------------------------
all_lights  = find(light_type ~= 0); % Select all ilumination indexes
white_light = find(light_type == 1); % Select just white light ilumination
nbi_light   = find(light_type == 2); % Select just NBI ilumination

%--------------------------------------------------------------------------
% Database size initial settings
%--------------------------------------------------------------------------
samples_inds = all_lights;                     % Use it as a pivot to select the ilumination
classes_num  = max(class_label(samples_inds)); % Get the number of used classes

clear all_lights white_light nbi_light;

%--------------------------------------------------------------------------
% Neural networks settings
%--------------------------------------------------------------------------
L_mlp         = [7, 5]; % 1-by-P vector of neurons for each layer
L_rbf         = 5;      % Number of neurons for the hidden-layer
learning_rate = 0.01;   % Learning rate to be used in weights adjusts
epochs        = 500;    % Max number of epochs
err           = 1e-7;   % Mean squared error goal for both neural networks

%% Adjust and Balance the Database and create Label Vectors
%
%  The code bellow creates a binary label representation where the correct
%  class for a sample is set to 1 and other classes indexes are set to 0.
%
%  The dataset is, also, balanced in order to keep the same number of
%  samples for every class.
%

%--------------------------------------------------------------------------
% The following lines separate all samples by their classes in order to
% balance and discard extra samples from bigger classes. This task is made
% in order to keep all classes with the same number of samples
%--------------------------------------------------------------------------
classified_samples  = {};
classes_upper_bound = size(features(:, samples_inds), 2);

for class = 1:classes_num
    classified_samples{class} = find(class_label(samples_inds) == class);
    
    if (size(classified_samples{class}, 2) < classes_upper_bound)
        classes_upper_bound = size(classified_samples{class}, 2);
    end
end

samples_num = classes_upper_bound; % The samples num is redefined

clear classes_upper_bound;

%--------------------------------------------------------------------------
% Since we know which is the smallest class and the number of samples it
% has (upper bound), it is time to cut off other samples class from classes
% which the number of samples is higher than the upper bound.
%--------------------------------------------------------------------------
rng('shuffle'); % Just to make sure the seed will be randomly chosen
selected_samples_inds = zeros(classes_num, samples_num);

%--------------------------------------------------------------------------
% Select the same number of samples for each class
%--------------------------------------------------------------------------
for class = 1:classes_num
    class_samples_num = size(classified_samples{class}, 2);
    class_samples_inds = randperm(class_samples_num, samples_num);
    selected_samples_inds(class, :) = classified_samples{class}(class_samples_inds);
end

clear class_samples_num;

%--------------------------------------------------------------------------
% Compose the database with just the randomly chosen samples and create the
% label vectors
%--------------------------------------------------------------------------
X = zeros(size(features, 1), classes_num * samples_num);
D = zeros(classes_num, classes_num * samples_num);
samples_count = zeros(classes_num, 1);
database_ind  = 1;

for sample = 1:samples_num
    for class = 1:classes_num
        sample_index           = selected_samples_inds(class, sample);
        X(:, database_ind)     = features(:, sample_index);
        D(class, database_ind) = 1;
        samples_count(class)   = samples_count(class) + 1;
        database_ind = database_ind + 1;
    end
end

samples_num = samples_num * classes_num; % Samples num is redefined to represent the whole database

clear class database_ind samples_count samples_inds selected_samples_inds;

%% Build Neural Networks
%
%  The functions bellow build the neural networks with global parameters,
%  specified above.
%
mlp = NeuralNetworks.MLP(X, D, L_mlp, learning_rate, epochs, err);
rbf = NeuralNetworks.RBF(X, D, L_rbf, learning_rate, epochs, err);

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

%--------------------------------------------------------------------------
% Variables bellow store the accuracies for each trial in both LOO and
% KFold validations.
%--------------------------------------------------------------------------
mlp_accuracies = zeros(2, trials);
rbf_accuracies = zeros(2, trials);

%--------------------------------------------------------------------------
% The following variables store the confusion matrices for each LOO and
% KFold validations. Each confusion matrix is already a mean matrix of the
% validation internal executions.
%--------------------------------------------------------------------------
mlp_confusions = zeros(classes_num, classes_num, trials);
rbf_confusions = zeros(classes_num, classes_num, 2, trials);

%--------------------------------------------------------------------------
% "loo_ind" and "kfold_ind" are just names to the indexes. It is good for
% better understanding of the attributions returned from validation methods
%--------------------------------------------------------------------------
loo_ind   = 1;
kfold_ind = 2;

for t = 1:trials
    fprintf("Trial %2d/%2d", t, trials);
    fprintf("\n--------------------------");
    
    fprintf("\n\tMLP - LOO: ");
    [accuracy, confusion]            = Validations.LOO(mlp, X, D);
    mlp_accuracies(loo_ind, t)       = accuracy;
    mlp_confusions(:, :, loo_ind, t) = confusion;
    fprintf("%.4f", mlp_accuracies(loo_ind, t));
    
    fprintf("\n\tRBF - LOO: ");
    [accuracy, confusion]            = Validations.LOO(rbf, X, D);
    rbf_accuracies(loo_ind, t)       = accuracy;
    rbf_confusions(:, :, loo_ind, t) = confusion;
    fprintf("%.4f", rbf_accuracies(loo_ind, t));
    
    fprintf("\n\tMLP - %d-Fold: ", k);
    [accuracy, confusion]              = Validations.KFold(mlp, X, D, k);
    mlp_accuracies(kfold_ind, t)       = accuracy;
    mlp_confusions(:, :, kfold_ind, t) = confusion;
    fprintf("%.4f", mlp_accuracies(kfold_ind, t));
    
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

