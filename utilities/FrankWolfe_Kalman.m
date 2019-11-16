function [G, Q_star, obj, res] = FrankWolfe_Kalman(mu, Sigma, rho, x_dim, opts)
    % Parameters
    n = x_dim;
    iter_max = 1000;
    bi_tol = 1e-8;
    tol = 1e-4;
    verbose = false;
    if nargin == 5
        if isfield(opts, 'iter_max')
            iter_max = opts.iter_max;
        end
        if isfield(opts, 'bi_tol')
            bi_tol = opts.bi_tol;
        end
        if isfield(opts, 'tol')
            tol = opts.tol;
        end
        if isfield(opts, 'verbose')
            verbose = opts.verbose;
        end
    end

    % Auxilary functions
    G_ = @(S) S(1:n, n+1:end)/(S(n+1:end, n+1:end));
    f_ = @(S, G) trace(S(1:n, 1:n) - G * S(n+1:end, 1:n));
    grad_f_ = @(G) [eye(n), -G]' * [eye(n), -G];
    vec = @(x) x(:);

    % Initialize    
    S = Sigma;
    obj = zeros(iter_max,1);
    res = zeros(iter_max,1);

    % Return Bayes estimator if rho is zero
    if rho == 0
        G = G_(Sigma);
        Q_star.Sigma = Sigma;
        Q_star.mu = mu;
        return
    end

    % Print the caption
    if verbose
        fprintf('%s\n', repmat('*', 1, 22));
        fprintf('Frank-Wolfe''s method \n');
        fprintf('%s\n', repmat('*', 1, 22));
        fprintf('  Iter|  Objective \n');
        fprintf('%s\n', repmat('*', 1, 22));
    end

    % The main loop
    for iter = 0 : iter_max

        % Printing the objective values
        G = G_(S);
        obj_current = f_(S, G);
        if mod(iter, fix(iter_max/10)) == 0 && verbose  
            fprintf('%6d| %5.4e\n', iter, obj_current);
        end

        % Computing partial derivitive & solve the algebraic equation
        D = grad_f_(G);
        L = my_bisection(Sigma, D, rho, bi_tol);

        % Check the optimality condition
        res_current = abs(vec(L-S)'* vec(D));
        if res_current / obj_current < tol
            break
        end

        % Updating the current solution
        alpha = 2 / (2 + iter);
        S = (1-alpha) * S + alpha * L;
        
        % Store results
        if iter > 0
            obj(iter) = obj_current;
            res(iter) = abs(vec(L-S)'* vec(D));
        end

    end

    if iter < iter_max
        obj = obj(1:iter);
        res = res(1:iter);
    end

    Q_star.Sigma = S;
    Q_star.mu = mu;

end

function L = my_bisection(Sigma, D, rho, bi_tol)
%   Bisection algorithm - MATLAB implementation of Algorithm 1
%
%   Syntax: L = my_bisection(Sigma, D, rho, bi_tol)
%   my_bisection() computes the solution to the subproblem in the Frank-Wolfe algorithm.
%
%   Sigma:  Covariance matrix of the prior distribution
%   D:      The gradient of the objective function at an iteration
%   rho:    Wasserstein ambiguity size
%   bi_tol: Bisection tolerance value

    d = size(D,1);

    % Auxilary functions
    h = @(inv_D) rho^2 - sum(sum(Sigma .* (eye(d) - inv_D)^2));
    vec = @(x) x(:);

    % Finding the bisection intervals
    [v_1, lambda_1] = eigs(D,1);
    LB = lambda_1 * (1 + sqrt(v_1'*Sigma*v_1)/rho);
    UB = lambda_1 * (1 + sqrt(trace(Sigma))/rho);

    % The main loop
    while true
        gamma = (LB + UB)/2;

        D_inv = gamma * inv(gamma*eye(d) - D);
        L = D_inv * Sigma * D_inv;
        h_val = h(D_inv);

        if h_val < 0
            LB = gamma;
        else
            UB = gamma;
        end

        Delta = gamma * (rho^2 - trace(Sigma)) + gamma * vec(D_inv)'*vec(Sigma) - vec(L)'*vec(D);

        if (h_val >= 0) && (Delta < bi_tol)
            break
        end

    end
end