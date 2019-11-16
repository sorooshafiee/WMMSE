function L = linear_oracle(nabla, var_cov, covsa, rho, delta, oracle)
% Compute the solution to the linear maximization problem
%           max   < L, nabla'*nabla >
%           s.t   W(L, Sigma) <= rho,
% where W(., .) is the Wasserstein distance between two covariance matrix
% and D = nabla'*nabla is the first order derivative.

    D = nabla'*nabla;
    if rho == 0
        L = covsa;
    else
        switch oracle
            case 'inv'
                L = linear_oracle_inv(D, var_cov, covsa, rho, delta);
            case 'eig'
                L = linear_oracle_eig(D, var_cov, covsa, rho, delta);
            case 'svd'
                L = linear_oracle_svd(nabla, var_cov, covsa, rho, delta);
            otherwise
                L = dual_solution(D, covsa, rho, 1e-12);
        end
    end
    
end