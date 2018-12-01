function network = RBF(X, D, L, learning_rate, epochs, err)
%RBF Creates a Radial Basis Function Neural Network
%   This function creates a configured a RBF Neural Network with the
%   following parameters:
%
%   X                M-by-N matrix of M features and N input vectors
%   D                1-by-N vector of labels for all N input vectors
%   L                Number of neurons for the hidden-layer
%   learning_rate    The learning rate for the neural network
%   epochs           Upper bound of epochs for the neural network
%   err              The target mean square error
%

    network = feedforwardnet(L, 'trainlm');
    
    network.layers{1}.transferFcn = 'radbas';
    network.layers{2}.transferFcn = 'purelin';
    network.trainParam.lr         = learning_rate;
    network.trainParam.epochs     = epochs;
    network.trainParam.goal       = err;
    network.trainParam.showWindow = false;
    network.trainParam.max_fail   = 1000;
    network.divideFcn             = '';
    network.name                  = 'rbf';
    
    fprintf("BUILDING RADIAL BASIS FUNCTION");
    fprintf("\n------------------------------");
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

