function [accuracy, confusion] = KFold(net, X, D, k)
%KFOLD Get the mean accuracy with a K-Fold cross-validation
%   To get the mean accuracy of a neural network with the K-fold validation
%   you just need to pass the neural network, the database, labels and a
%   value for K.
%
    
    indexes     = crossvalind('Kfold', size(X, 2), k);
    classes_num = max(D);
    accuracies  = zeros(1, k);
    confusion   = zeros(classes_num);
    
    for i = 1:k
        ind_train = find(indexes ~= i);
        ind_test  = find(indexes == i);
        
        X_train = X(:, ind_train);
        D_train = D(ind_train);
        
        X_test = X(:, ind_test);
        D_test = D(ind_test);
        
        L = ones(1, length(ind_test));               % Classification lower bound
        U = ones(1, length(ind_test)) * classes_num; % Classification upper bound
        
        net   = train(net, X_train, D_train);
        D_out = round(sim(net, X_test));
        D_out = min(U, max(L, D_out));
        
        accuracies(i) = sum(D_out == D_test) / length(D_test);
        
        for j = 1:length(D_test)
            confusion(D_test(j), D_out(j)) = confusion(D_test(j), D_out(j)) + 1;
        end
    end
    
    accuracy  = mean(accuracies);
end

