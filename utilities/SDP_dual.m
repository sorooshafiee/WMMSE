function [A_star, b_star, obj, diagnosis] = SDP_dual(mu_x, cov_x, rho_x, mu_w, cov_w, rho_w, H, ops)
% SDP_MMSE - Solving problem (8)
%
% Syntax: [A_star, b_star, obj, diagnosis] = SDP_dual(mu_x, cov_x, rho_x, mu_w, cov_w, rho_w, H)
%
% Long description

    % Define auxiliary variable
    n = length(mu_x);
    m = length(mu_w);
    cov_x_sqrt = sqrtm(cov_x);
    cov_w_sqrt = sqrtm(cov_w);    

    % Define decision variables
    S_x = sdpvar(n, n);
    V_x = sdpvar(n, n);
    U = sdpvar(n, n);
    S_w = sdpvar(m, m);
    V_w = sdpvar(m, m);

    % Declare objective function
    objective = trace(S_x) - trace(U);

    % Define constraints
    constraints = [S_x >= 0, V_x >= 0, U >= 0, S_w >= 0, V_w >= 0, ...
                   [cov_x_sqrt * S_x * cov_x_sqrt, V_x; V_x, eye(n)] >= 0, ...
                   [cov_w_sqrt * S_w * cov_w_sqrt, V_w; V_w, eye(m)] >= 0, ...
                   trace(S_x + cov_x - 2 * V_x) <= rho_x^2, ...
                   trace(S_w + cov_w - 2 * V_w) <= rho_w^2, ...
                   [U, S_x * H'; H * S_x, H * S_x * H' + S_w] >= 0];

    % Solving the Optimization Problem           
    diagnosis = optimize(constraints, -objective, ops);
    var_cov_x = value(S_x);
    var_cov_w = value(S_w);
    A_star = var_cov_x * H' / (H * var_cov_x * H' + var_cov_w);
    b_star = mu_x - A_star * (H * mu_x + mu_w);
    obj = value(objective);
end