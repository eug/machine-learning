function [theta, J_history, Theta_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    t_theta = theta;
    for j = 1:length(theta) 
        x = X(:,j);       % get x^i_j
        h = X * t_theta;  % hipothesis
        error = h - y;    % hipothesis error
        delta = sum( error .* x ) / m;
        theta(j) =  t_theta(j) - alpha * delta;
    end

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end

end
