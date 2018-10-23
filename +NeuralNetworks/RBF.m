function network = RBF(X, D, err, spread, max_q, increment_q)
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
    network = newrb(X, D, err, spread, max_q, increment_q);   
end

