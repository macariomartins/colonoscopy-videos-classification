function network = MLP(X, D, L, learning_rate, epochs, err)
%MLP Creates a Multi-Layer Perceptron Neural Network
%parameters:
%   This function creates a configured MLP Neural Network with the
%   following parameters:
%
%   X   P-by-N matrix of P features and N input vectors
%   D   R-by-N matrix of R labels and N target class vectors
%   Q   Row vector of one or more hidden layer sizes.
%

    network = feedforwardnet(L, 'trainlm');
    
    network.name                  = 'mlp';
    network.layers{1}.transferFcn = 'tansig';
    network.layers{2}.transferFcn = 'tansig';
    network.trainParam.lr         = learning_rate;
    network.trainParam.epochs     = epochs;
    network.trainParam.goal       = err;
    network.trainParam.showWindow = false;
    network.trainParam.max_fail   = 1000;
    network.divideFcn             = '';
    
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

