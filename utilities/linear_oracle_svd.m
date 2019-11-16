function L = linear_oracle_svd(nabla, var_cov, covsa, rho, delta)
% Compute the solution to the linear maximization problem
%           max   < L, nabla'*nabla >
%           s.t   W(L, Sigma) <= rho,
% where W(., .) is the Wasserstein distance between two covariance matrix
% and nabla'*nabla is the first order derivative.

    % Auxilary function
    vec = @(x) x(:);
    D = nabla' * nabla;
    
    % SVD decomposition
    [~, S, V] = svds(nabla, min(size(nabla)), 'la');

    % Eigenvalue bisection
    d = diag(S).^2; d = d(:)';
    M1 = covsa * V;
    M2 = V' * M1;
    g1 = @(x) (1 ./ (x - d));
    g2 = @(x) (d ./ (x - d)).^2;    
    LB = d(1) * (1 + (sqrt(V(:,1)' * covsa * V(:,1)) / rho));
    UB = d(1) * (1 + (sqrt(trace(covsa)) / rho));
    offset_D = vec(var_cov)' * vec(D);
    phi = @(x) x * (rho^2 - trace(covsa)) + x.^2 * vec(V .* g1(x))' * M1(:) - offset_D;
    d_phi = @(x) rho^2 - vec(V .* g2(x))' * M1(:);
    Delta = @(x) (x.^2 * trace((g1(x)' .* M2 .* g1(x)) .* d) - offset_D) / phi(x);
    while true
        dual = (LB + UB)/2;
        d_phi_val = d_phi(dual);
        if d_phi_val < 0 
            LB = dual;
        else
            UB = dual;
        end
        if (d_phi_val >= 0) && (Delta(dual) > delta || UB - LB <= 1e-12)
            break
        end
    end
    
    % Reconstruct the solution from eigenvalues
    L_tilde = V * diag(1 ./ (dual - d)) * V';
    L = dual^2 * L_tilde * covsa * L_tilde;
    
end