function accuracy = KFold(net, X, D, k)
%KFOLD Get the mean accuracy with a K-Fold cross-validation
%   To get the mean accuracy of a neural network with the K-fold validation
%   you just need to pass the neural network, the database, labels and a
%   value for K.
%
    
    indexes    = crossvalind('Kfold', size(X, 2), k);
    accuracies = zeros(1, k);
    
    for i = 1:k
        ind_train = find(indexes ~= i);
        ind_test  = find(indexes == i);
        
        X_train = X(:, ind_train);
        D_train = D(ind_train);
        
        X_test = X(:, ind_test);
        D_test = D(ind_test);
        
        net = train(net, X_train, D_train);
        D_output = round(sim(net, X_test));
        
        accuracies(i) = sum(D_output == D_test) / length(D_test);
    end
    
    accuracy = mean(accuracies);
end

