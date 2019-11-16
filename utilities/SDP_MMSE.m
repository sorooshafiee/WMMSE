function [A_star, b_star, obj, diagnosis] = SDP_MMSE(mu_x, cov_x, rho_x, mu_w, cov_w, rho_w, H, verbose)
% SDP_MMSE - Solving problem (8)
%
% Syntax: [A_star, b_star, obj, diagnosis] = SDP_MMSE(mu_x, cov_x, rho_x, mu_w, cov_w, rho_w, H)
%
% Long description

    % Define auxiliary variable
    n = length(mu_x);
    m = length(mu_w);
    L_x = chol(cov_x, 'lower');
    L_w = chol(cov_w, 'lower');

    % Initialization
    ops = sdpsettings('solver', 'mosek', 'verbose', verbose);

    % Define decision variables
    dual_x = sdpvar(1,1);
    dual_w = sdpvar(1,1);
    A = sdpvar(n,m,'full');
    U_x = sdpvar(n,n);
    V_x = sdpvar(n,n);
    U_w = sdpvar(m,m);
    V_w = sdpvar(m,m);

    % Declare objective function
    objective = dual_x * (rho_x^2 - trace(cov_x)) + dual_w * (rho_w^2 - trace(cov_w)) + trace(U_x) + trace(U_w);

    % Define constraints
    K = A * H - eye(n);
    constraints = [dual_x >= 0, dual_w >= 0, U_x >= 0, V_x >= 0, U_w >= 0, V_w >= 0, ...
                   [U_x, dual_x * L_x'; dual_x * L_x, V_x] >= 0, ...
                   [dual_x * eye(n) - V_x, K'; K, eye(n)] >= 0, ...
                   [U_w, dual_w * L_w'; dual_w * L_w, V_w] >= 0, ...
                   [dual_w * eye(m) - V_w, A'; A, eye(m)] >= 0];

    % Solving the Optimization Problem           
    diagnosis = optimize(constraints, objective, ops);
    A_star = value(A);
    b_star = mu_x - A_star * (H * mu_x + mu_w);
    obj = value(objective);    
end