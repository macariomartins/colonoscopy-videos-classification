function network = RBF(X, D, L, learning_rate, epochs, err)
%RBF Creates a Radial Basis Function Neural Network
%   This function creates and train a RBF Neural Network with the following
%   parameters:
%
%   X           P-by-N matrix of P features and N input vectors
%   D           R-by-N matrix of R labels and N target class vectors
%   err         Mean squared error goal for both neural networks
%   spread      Spread of radial basis functions
%   max_q       Maximum number of neurons
%   increment_q Number of neurons to add between displays
%

%     network = newrb(X, D, err, spread, max_q, increment_q);   

    network = feedforwardnet(L, 'trainlm');
    
    network.layers{1}.transferFcn = 'radbas';
    network.layers{2}.transferFcn = 'purelin';
    network.trainParam.lr         = learning_rate;
    network.trainParam.epochs     = epochs;
    network.trainParam.goal       = err;
    network.trainParam.showWindow = false;
    network.trainParam.max_fail   = 1000;
    network.divideFcn             = '';
    
    fprintf("BUILDING RADIAL BASIS FUNCTION WITH PARAMETERS:");
    fprintf("\n> X(%dx%d)", size(D, 1), size(D, 2));
    fprintf("\n> D(%dx%d)", size(D, 1), size(D, 2));
    fprintf("\n> Neurons layers: ");
    
    for i = 1:size(L, 2)
        fprintf("\n\tLayer %d: %d neurons", i, L(i));
    end
    
    fprintf("\n> Learning rate: %f", learning_rate);
    fprintf("\n> Max. Epochs: %d", epochs);
    fprintf("\n> Error: %f\n\n", err);
    
    tic;
    network = train(network, X, D);
    toc;
    
    printf("\n\n");
end

