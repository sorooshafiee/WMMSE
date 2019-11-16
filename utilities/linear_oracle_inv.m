function L = linear_oracle_inv(D, var_cov, covsa, rho, delta)
% Compute the solution to the linear maximization problem
%           max   < L, D >
%           s.t   W(L, Sigma) <= rho,
% where W(., .) is the Wasserstein distance between two covariance matrix
% and D is the first order derivative.

    d = size(D,1);

    % Auxilary functions
    vec = @(x) x(:);    

    % Finding the bisection intervals
    [v_1, lambda_1] = eigs(D,1);
    LB = lambda_1 * (1 + sqrt(v_1'*covsa*v_1)/rho);
    UB = lambda_1 * (1 + sqrt(trace(covsa))/rho);
    offset_D = vec(var_cov)' * vec(D);

    % The main loop
    while true
        gamma = (LB + UB)/2;

        D_inv = gamma * inv(gamma*eye(d) - D);
        L = D_inv * covsa * D_inv;
        phi = gamma * (rho^2 - trace(covsa)) + gamma * vec(D_inv)' * vec(covsa) - offset_D;
        d_phi = rho^2 - vec(covsa)' * vec((eye(d) - D_inv)^2);

        if d_phi < 0
            LB = gamma;
        else
            UB = gamma;
        end

        Delta = (vec(L)'*vec(D) - offset_D) / phi;

        if (d_phi >= 0) && (Delta > delta || UB - LB <= 1e-12)
            break
        end

    end
    
end