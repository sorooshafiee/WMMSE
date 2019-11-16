clc
clear
rng(100);
addpath(genpath('utilities'));

n = 50;
n_samples = 100;
run_count = 100;
opts.iter_max = 1000;
opts.tol = 1e-4;
rho = [0, (1:9)*1e-1, 1:10];
H = eye(n);

% Auxilary functions
MSE = @(A, cov_xx, cov_ww) trace((eye(n) - A * H) * cov_xx * (eye(n) - A * H)' + A * cov_ww * A');
A_opt = @(cov_xx, cov_ww) (cov_xx * H') / (H * cov_xx * H' + cov_ww);


MSE_star = NaN(run_count,1);
MSE_0 = NaN(run_count,1);
MSE_rob_kalman = NaN(run_count, length(rho));
MSE_rob = NaN(run_count, length(rho), length(rho));

% estimation based or random generation
is_est = true;

for r = 1 : run_count
    fprintf('%s\n', repmat('*', 1, 22));
    fprintf('iteration %d\n', r);
    
    A_star = randn(n);
    [R_A_star, ~] = eig(A_star + A_star');
    lambda_x_star = 1 + 4 * rand(n,1);
    cov_x_star = R_A_star * diag(lambda_x_star) * R_A_star';

    B_star = randn(n);
    [R_B_star, ~] = eig(B_star + B_star');
    lambda_w_star = 1 + rand(n,1);
    cov_w_star = R_B_star * diag(lambda_w_star) * R_B_star';
    
    if is_est
        R_x = mvnrnd(zeros(n,1), cov_x_star, n_samples);
        R_w = mvnrnd(zeros(n,1), cov_w_star, n_samples);
        cov_x = cov(R_x);
        cov_w = cov(R_w);
    else
        A = randn(n);
        [R_A, ~] = eig(A + A');
        lambda_x = 1 + 4 * rand(n,1);
        cov_x = R_A * diag(lambda_x) * R_A';
        B = randn(n);
        [R_B, ~] = eig(B + B');
        lambda_w = 1 + rand(n,1);
        cov_w = R_B * diag(lambda_w) * R_B';        
    end
    
    A_0 = A_opt(cov_x, cov_w);
    MSE_0(r) = MSE(A_0, cov_x_star, cov_w_star);

    A_star = A_opt(cov_x_star, cov_w_star);
    MSE_star(r) = MSE(A_star, cov_x_star, cov_w_star);
    
    for i = 1 : length(rho)
        rho_xy = rho(i);
        cov_xy = [cov_x, cov_x*H'; H*cov_x, H * cov_x * H' + cov_w];
        A_rob_kalman = FrankWolfe_Kalman(zeros(2 * n,1), cov_xy, rho_xy, n, opts);
        MSE_rob_kalman(r, i) = MSE(A_rob_kalman, cov_x_star, cov_w_star);
        for j = 1 : length(rho)
            rho_x = rho(i);
            rho_w = rho(j);
            A_rob = FrankWolfe(zeros(n,1), cov_x, rho_x, zeros(n,1), cov_w, rho_w, eye(n), opts);
            MSE_rob(r,i,j) = MSE(A_rob, cov_x_star, cov_w_star);
        end
    end
end
save toy_data2 MSE_rob MSE_rob_kalman MSE_star MSE_0 rho
%% Plot the results
load toy_data2
rho = rho(2:end);
font_size = 24;
colors = [0, 0.45, 0.75; 0.85, 0.325, 0.01; 0.925, 0.70, 0.125];
prc = 0;
alphaa = 0.1;

fig = figure;
set(fig, 'Units', 'normalized', 'Position', [0.35, 0.25, 0.4, 0.55])
hold on
plot_with_shade(rho, transpose(MSE_rob_kalman(:, 2:end) - repmat(MSE_star,[1,length(rho)])), prc, alphaa, colors(1,:));
grid on
set(gca, 'XScale', 'log', 'FontSize', font_size - 6);
xlabel('$\rho$', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Regret', 'Interpreter', 'latex', 'FontSize', font_size);
saveas(gcf,'fig2-a','svg')

fig = figure;
set(fig, 'Units', 'normalized', 'Position', [0.35, 0.25, 0.4, 0.55])
hold on
MSE_rob_x = squeeze(MSE_rob(:,1,2:end));
plot_with_shade(rho, transpose(MSE_rob_x - repmat(MSE_star,[1,length(rho)])), prc, alphaa, colors(1,:));
grid on
set(gca, 'XScale', 'log', 'FontSize', font_size - 6);
xlabel('$\rho_x$', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Regret', 'Interpreter', 'latex', 'FontSize', font_size);
saveas(gcf,'fig2-b','svg')

fig = figure;
set(fig, 'Units', 'normalized', 'Position', [0.35, 0.25, 0.4, 0.55])
hold on
MSE_rob_w = squeeze(MSE_rob(:,2:end,1));
plot_with_shade(rho, transpose(MSE_rob_w - repmat(MSE_star,[1,length(rho)])), prc, alphaa, colors(1,:));
grid on
set(gca, 'XScale', 'log', 'FontSize', font_size - 6);
xlabel('$\rho_x$', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Regret', 'Interpreter', 'latex', 'FontSize', font_size);
saveas(gcf,'fig2-c','svg')

fig = figure;
set(fig, 'Units', 'normalized', 'Position', [0.35, 0.25, 0.4, 0.55])
hold on
p1 = plot_with_shade(rho, transpose(MSE_rob_kalman(:, 2:end) - repmat(MSE_star,[1,length(rho)])), prc, alphaa, colors(1,:));
p2 = plot_with_shade(rho, transpose(MSE_rob_x - repmat(MSE_star,[1,length(rho)])), prc, alphaa, colors(2,:));
p3 = plot_with_shade(rho, transpose(MSE_rob_w - repmat(MSE_star,[1,length(rho)])), prc, alphaa, colors(3,:));
grid on
set(gca, 'XScale', 'log', 'FontSize', font_size - 6);
xlabel('$\rho$', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Regret', 'Interpreter', 'latex', 'FontSize', font_size);
lgd = legend([p1, p2, p3], 'Unstructured WMMSE', 'Structured WMMSE with $\rho_w=0$', ...
                           'Structured WMMSE with $\rho_x=0$', 'Location', 'northwest');
set(lgd,'Interpreter','latex', 'FontSize', font_size-6);
saveas(gcf,'fig2-all','svg')

fig = figure;
set(fig, 'Units', 'normalized', 'Position', [0.35, 0.25, 0.4, 0.55])
[xx, ww] = meshgrid(rho,rho);
result_rob = squeeze(mean(MSE_rob(:, 2:end, 2:end), 1)) - mean(MSE_star);
surf(xx, ww, result_rob, 'FaceColor', 'interp')
set(gca, 'XScale', 'log', 'YScale', 'log', 'FontSize', font_size - 6);
xlabel('$\rho_x$', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('$\rho_w$', 'Interpreter', 'latex', 'FontSize', font_size);
zlabel('Regret', 'Interpreter', 'latex', 'FontSize', font_size);
saveas(gcf,'fig2-3d','svg')