function L = dual_solution(D, covsa, rho, tol)
% Bisection algorithm for Solving the algebraic equations
%
% Syntax: p = bisection(f, a, b)
%
% Long description

    % Auxilary function
    vec = @(x) x(:);
    
    % Eigen decomposition
    [V, Lambda] = eigs((D+D')/2, length(D), 'la');

    % Eigenvalue bisection
    d = diag(Lambda)';
    M1 = covsa * V;
    g2 = @(x) (d ./ (x - d)).^2;    
    LB = d(1) * (1 + (sqrt(V(:,1)' * covsa * V(:,1)) / rho));
    UB = d(1) * (1 + (sqrt(trace(covsa)) / rho));
    d_phi = @(x) rho^2 - vec(V .* g2(x))' * M1(:);
    dual = my_bisection(d_phi, LB, UB, tol);
    L_tilde = V * diag(1 ./ (dual - d)) * V';
    L = dual^2 * L_tilde * covsa * L_tilde;
end