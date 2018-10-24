function accuracy = LOO(net, X, D)
%LOO Get the mean accuracy with the Leave-One-Out method cross-validation
%   To get the mean accuracy of a neural network with the Leave-One-Out
%   validation you just need to pass the neural network, the database and
%   labels. Since LOO is a special case of K-fold, where K is equal to the
%   database size, we will just reuse the K-fold function.

    accuracy = Validations.KFold(net, X, D, size(X, 2));
end

