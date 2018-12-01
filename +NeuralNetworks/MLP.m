function network = MLP(X, D, L, learning_rate, epochs, err)
%MLP Creates a Multi-Layer Perceptron Neural Network
%parameters:
%   This function creates a configured MLP Neural Network with the
%   following parameters:
%
%   X                M-by-N matrix of M features and N input vectors
%   D                1-by-N vector of labels for all N input vectors
%   L                1-by-P vector of neurons for each layer
%   learning_rate    The learning rate for the neural network
%   epochs           Upper bound of epochs for the neural network
%   err              The target mean square error
%

    network = feedforwardnet(L, 'trainlm');
    
    for l = 1:length(network.layers)-1
        network.layers{l}.transferFcn = 'tansig';
    end
    
    network.layers{length(network.layers)}.transferFcn = 'purelin';
    network.trainParam.lr         = learning_rate;
    network.trainParam.epochs     = epochs;
    network.trainParam.goal       = err;
    network.trainParam.showWindow = false;
    network.trainParam.max_fail   = 1000;
    network.divideFcn             = '';
    network.name                  = 'mlp';
    
    fprintf("BUILDING MULTI LAYER PERCEPTRON");
    fprintf("\n-------------------------------");
    fprintf("\n> X(%dx%d)", size(X, 1), size(X, 2));
    fprintf("\n> D(%dx%d)", size(D, 1), size(D, 2));
    fprintf("\n> Neurons layers: ");
    
    for i = 1:size(L, 2)
        fprintf("\n\tLayer %d: %d neurons", i, L(i));
    end
    
    fprintf("\n> Learning rate: %f", learning_rate);
    fprintf("\n> Max. Epochs: %d", epochs);
    fprintf("\n> Error: %f\n\n", err);
end

