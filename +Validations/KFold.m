function [accuracy, confusion] = KFold(net, X, D, k)
%KFOLD Get the mean accuracy with a K-Fold cross-validation
%   To get the mean accuracy of a neural network with the K-fold validation
%   you just need to pass the neural network, the database, labels and a
%   value for K.
%

    fold_size    = floor(size(X, 2) / k);
    classes_num  = size(D, 1);
    samples_num  = size(D, 2);
    start_points = 1:fold_size:samples_num;
    accuracies   = zeros(1, k);
    confusion    = zeros(classes_num);
    
    for i = 1:k
        ind_test  = start_points(i):start_points(i)+fold_size-1;
        ind_train = setdiff(1:samples_num, ind_test);
        
        X_train = X(:, ind_train);
        D_train = D(:, ind_train);
        
        X_test = X(:, ind_test);
        
        net = train(net, X_train, D_train);
        Y   = net(X_test);
        
        [~, D_test] = max(D(:, ind_test));
        [~, D_out ] = max(Y);
        
        accuracies(i) = sum(D_out == D_test) / length(ind_test);
        
        for j = 1:length(ind_test)
            confusion(D_test(j), D_out(j)) = confusion(D_test(j), D_out(j)) + 1;
        end
    end
    
    accuracy  = mean(accuracies);
end

