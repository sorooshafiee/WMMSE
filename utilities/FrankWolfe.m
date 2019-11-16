function [A_star, b_star, obj, res] = FrankWolfe(mu_x, cov_x, rho_x, mu_w, cov_w, rho_w, H, opts)
% Frank-Wolfe algorithm for Solving problem (17)
%
% Syntax: [A_star, b_star, obj] = FrankWolfe(mu_x, cov_x, rho_x, mu_w, cov_w, rho_w, H)
%
% Long description

    % Parameters
    iter_max = 1000;
    delta = 0.95;
    tol = 1e-5;
    tau = 2;
    mu = 0.5;
    oracle = '';
    step_size = 'full_adaptive';
    verbose = false;    
    if isfield(opts, 'iter_max')
        iter_max = opts.iter_max;
    end
    if isfield(opts, 'delta')
        delta = opts.delta;
    end
    if isfield(opts, 'tol')
        tol = opts.tol;
    end
    if isfield(opts, 'tau')
        tau = opts.tau;
    end
    if isfield(opts, 'mu')
        mu = opts.mu;
    end
    if isfield(opts, 'oracle')
        oracle = opts.oracle;
    end
    if isfield(opts, 'step_size')
        step_size = opts.step_size;
    end
    if isfield(opts, 'verbose')
        verbose = opts.verbose;
    end

    % Print the caption
    if verbose
        fprintf('%s\n', repmat('*', 1, 22));
        fprintf(strcat(step_size, ' Frank-Wolfe''s method \n'));
        fprintf('%s\n', repmat('*', 1, 22));
        fprintf('  Iter|  Objective \n');
        fprintf('%s\n', repmat('*', 1, 22));
    end

    % Initialize
    n = length(mu_x);
    m = length(mu_w);
    var_cov_x = cov_x;
    var_cov_w = cov_w;
    obj = zeros(iter_max,1);
    res = zeros(iter_max,1);

    % Auxiliary functions
    vec = @(x) x(:);
    G = @(x, w) H * x * H' + w;
    GH = @(x, w) G(x, w) \ H;
    f = @(x, w) trace(x - x * H' * GH(x, w) * x);
    g_x = @(x, w) eye(n) - x * H' * GH(x, w);
    g_w = @(x, w) (GH(x, w) * x)';    

    % Main loop
    lambda_min_w = min(eig(cov_w));
    lambda_max_x = (rho_x + sqrt(trace(cov_x)))^2;
    lambda_max_H = max(eig(H'*H));
    lambda_min_H = min(eig(H'*H));
    c = min(1 / lambda_min_H, lambda_max_H * lambda_max_x / lambda_min_w^2); 
    beta = 2 * (c + c * lambda_max_H + lambda_max_H) / lambda_min_w;
    beta_k = beta / tau^10;
    for iter = 0 : iter_max-1

        % Computing partial derivitives
        nabla_x = g_x(var_cov_x, var_cov_w);
        nabla_w = g_w(var_cov_x, var_cov_w);
        D_x = nabla_x' * nabla_x;
        D_w = nabla_w' * nabla_w;
        
        % Linear oracle solutions
        L_x = linear_oracle(nabla_x, var_cov_x, cov_x, rho_x, delta, oracle);
        L_w = linear_oracle(nabla_w, var_cov_w, cov_w, rho_w, delta, oracle);        

        % Check the objective value and the optimality condition
        obj(iter+1) = f(var_cov_x, var_cov_w);
        sub_opt_x = trace((L_x-var_cov_x)' * D_x);
        sub_opt_w = trace((L_w-var_cov_w)' * D_w);
        res(iter+1) = abs(sub_opt_x + sub_opt_w);
        if (mod(iter, 100) == 99 || iter == 0) && verbose  
            fprintf('%6d| %5.4e | %5.4e \n', iter+1, obj(iter+1), res(iter+1));
        end
        if res(iter+1) < tol
            break
        end
        
        % Compute the step size
        P_x = L_x - var_cov_x;
        P_w = L_w - var_cov_w;
        g_k = res(iter+1);        
        P_norm = vec(P_x)' * vec(P_x) + vec(P_w)' * vec(P_w);
        switch step_size
            case 'vanilla'
                eta_k = 2 / (iter+ 2);
            case 'adaptive'
                eta_k = min(g_k/(beta * P_norm), 1);
            case 'full_adaptive'
                % Line search for the best step size
                beta_k = beta_k * mu;
                while true
                    eta_k = min(g_k/(beta_k * P_norm), 1);
                    if f(var_cov_x + eta_k * P_x, var_cov_w + eta_k * P_w) >= ...
                            f(var_cov_x, var_cov_w) + eta_k * g_k - 0.5 * eta_k^2 * beta_k * P_norm
                        break
                    end
                    beta_k = beta_k * tau;
                end
        end        
        
        % Updating variables
        if rho_x > 0
            var_cov_x = (1-eta_k) * var_cov_x + eta_k * L_x;
        end
        if rho_w > 0
            var_cov_w = (1-eta_k) * var_cov_w + eta_k * L_w;
        end
    end
    A_star = var_cov_x * H' / (H * var_cov_x * H' + var_cov_w);
    b_star = mu_x - A_star * (H * mu_x + mu_w);
    if iter < iter_max-1
        obj = obj(1:iter+1);
        res = res(1:iter+1);
    else
        fprintf('The algorithm attains the duality gap %5.4e after %d iterations \n', res(end), iter_max);
    end
end