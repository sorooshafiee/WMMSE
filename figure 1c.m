clc
clear
rng(100);
addpath(genpath('utilities'));

d = 1000;
run_count = 10;
tol = 0;
verbose = true;
opts.iter_max = 1000;
opts.delta = 0.95;
opts.tol = tol;
opts.oracle = 'svd';
opts.verbose = verbose;

res_FW = NaN(opts.iter_max, run_count);
res_AFW = NaN(opts.iter_max, run_count);
res_FAFW = NaN(opts.iter_max, run_count);

for r = 1 : run_count

    rho_x = sqrt(d);
    rho_w = sqrt(d);
    fprintf('Running Iteration %d for d = %d \n', r, d);

    A = randn(d);
    [R_A, ~] = eig(A + A');
    lambda_x = 1 + 4 * rand(d,1);
    cov_x = R_A * diag(lambda_x) * R_A';

    B = randn(d);
    [R_B, ~] = eig(B + B');
    lambda_w = 1 + rand(d,1);
    cov_w = R_B * diag(lambda_w) * R_B';

    opts.step_size = 'vanilla';
    [~, ~, ~, res_FW(:,r)] = FrankWolfe(zeros(d,1), cov_x, rho_x, zeros(d,1), cov_w, rho_w, eye(d), opts);

    opts.step_size = 'adaptive';
    [~, ~, ~, res_AFW(:,r)] = FrankWolfe(zeros(d,1), cov_x, rho_x, zeros(d,1), cov_w, rho_w, eye(d), opts);

    opts.step_size = 'full_adaptive';
    [~, ~, ~, res_FAFW(:,r)] = FrankWolfe(zeros(d,1), cov_x, rho_x, zeros(d,1), cov_w, rho_w, eye(d), opts);

end
save convergence res_FW res_AFW res_FAFW opts
%%
load convergence
prc = 0;
alphaa = 0.1;
font_size = 20;
colors = [0, 0.45, 0.75; 0.85, 0.325, 0.01; 0.925, 0.70, 0.125];
fig = figure;
set(fig, 'Units', 'normalized', 'Position', [0.35, 0.25, 0.4, 0.55])
hold on
p1 = plot_with_shade(1:opts.iter_max, res_FW, prc, alphaa, colors(1,:));
p2 = plot_with_shade(1:opts.iter_max, res_AFW, prc, alphaa, colors(2,:));
p3 = plot_with_shade(1:opts.iter_max, res_FAFW, prc, alphaa, colors(3,:));
set(gca, 'XScale', 'log', 'YScale', 'log');
set(gca, 'FontSize', font_size - 2);
xlabel('# iterations', 'FontSize', font_size);
ylabel('Surrogate duality gap','FontSize', font_size)
ylim([1e-5, 1e4]);
grid on
lgd = legend([p1, p2, p3], 'Vanilla FW', 'Adaptive FW', 'Fully Adaptive FW', 'Location', 'southeast');
lgd.FontSize = font_size;
remove_border()
saveas(gcf, 'convergence', 'svg')