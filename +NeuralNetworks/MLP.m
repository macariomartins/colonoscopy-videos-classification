function network = MLP(X, D, Q, err)
%MLP Creates a Multi-Layer Perceptron Neural Network
%parameters:
%   This function creates and train a MLP Neural Network with the following
%   parameters:
%
%   X   P-by-N matrix of P features and N input vectors
%   D   R-by-N matrix of R labels and N target class vectors
%   Q   Row vector of one or more hidden layer sizes.
%
    classes_num = max(D);
    samples_num = size(D, 2);
    E = zeros(classes_num, samples_num);
    
    for i = 1:samples_num
        E(D(i), i) = 1;
    end
    
    network = feedforwardnet(Q, 'trainlm');
    network.trainParam.goal = err;
    network = train(network, X, E);
end

