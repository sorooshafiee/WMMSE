clc
clear
addpath(genpath('utilities'));

all_d = [(1:9)*10, (1:9)*100, 1000];
run_count = 10;
tol = 1e-3;
verbose = true;
opts.iter_max = 10000;
opts.delta = 0.95;
opts.tol = tol;
opts.oracle = 'svd';
opts.verbose = verbose;
ops = sdpsettings();
ops.solver = 'mosek';
ops.verbose = verbose;
ops.savesolveroutput = true;
ops.mosek.MSK_DPAR_INTPNT_CO_TOL_REL_GAP = tol;

time_FW = NaN(length(all_d), run_count);
iter_FW = NaN(length(all_d), run_count);
time_AFW = NaN(length(all_d), run_count);
iter_AFW = NaN(length(all_d), run_count);
time_FAFW = NaN(length(all_d), run_count);
iter_FAFW = NaN(length(all_d), run_count);
time_MSK = NaN(length(all_d), run_count);
iter_MSK = NaN(length(all_d), run_count);
enough_memory = true;

for i = 1 : length(all_d)
    for r = 1 : run_count
        d = all_d(i);
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
        
        t1 = cputime;
        opts.step_size = 'vanilla';
        [~, ~, obj, ~] = FrankWolfe(zeros(d,1), cov_x, rho_x, zeros(d,1), cov_w, rho_w, eye(d), opts);
        t2 = cputime;
        time_FW(i,r) = t2 - t1;
        iter_FW(i,r) = length(obj);
        
        t1 = cputime;
        opts.step_size = 'adaptive';
        [~, ~, obj, ~] = FrankWolfe(zeros(d,1), cov_x, rho_x, zeros(d,1), cov_w, rho_w, eye(d), opts);
        t2 = cputime;
        time_AFW(i,r) = t2 - t1;
        iter_AFW(i,r) = length(obj);
        
        t1 = cputime;
        opts.step_size = 'full_adaptive';
        [~, ~, obj, ~] = FrankWolfe(zeros(d,1), cov_x, rho_x, zeros(d,1), cov_w, rho_w, eye(d), opts);
        t2 = cputime;
        time_FAFW(i,r) = t2 - t1;
        iter_FAFW(i,r) = length(obj);

        if enough_memory && d < 100
            try
                t1 = cputime;
                [~, ~, obj, diagnosis] = SDP_dual(zeros(d,1), cov_x, rho_x, zeros(d,1), cov_w, rho_w, eye(d), ops);
                t2 = cputime;
                time_MSK(i,r) = t2 - t1;
                iter_MSK(i,r) = diagnosis.solveroutput.res.info.MSK_IINF_INTPNT_ITER;
            catch
                enough_memory = false;
            end
        end

    end
end
save final_results time_FW iter_FW time_AFW iter_AFW time_FAFW iter_FAFW time_MSK iter_MSK
%%
load final_results
prc = 0;
alphaa = 0.1;
font_size = 20;
colors = [0, 0.45, 0.75; 0.85, 0.325, 0.01; 0.925, 0.70, 0.125; 0.50, 0.20, 0.55];
fig = figure;
set(fig, 'Units', 'normalized', 'Position', [0.35, 0.25, 0.4, 0.55])
hold on
p1 = plot_with_shade(all_d, time_FW, prc, alphaa, colors(1,:));
p2 = plot_with_shade(all_d, time_AFW, prc, alphaa, colors(2,:));
p3 = plot_with_shade(all_d, time_FAFW, prc, alphaa, colors(3,:));
p4 = plot_with_shade(all_d, time_MSK, prc, alphaa, colors(4,:));
set(gca, 'XScale', 'log', 'YScale', 'log');
set(gca, 'FontSize', font_size - 2);
xlabel('Dimension', 'FontSize', font_size);
ylabel('Execution time (s)','FontSize', font_size)
grid on
lgd = legend([p1, p2, p3, p4], 'Vanilla FW', 'Adaptive FW', 'Fully Adaptive FW', 'MOSEK', 'Location', 'southeast');
lgd.FontSize = font_size;
remove_border()
saveas(gcf, 'time', 'svg')

fig = figure;
set(fig, 'Units', 'normalized', 'Position', [0.35, 0.25, 0.4, 0.55])
hold on
p1 = plot_with_shade(all_d, iter_FW, prc, alphaa, colors(1,:));
p2 = plot_with_shade(all_d, iter_AFW, prc, alphaa, colors(2,:));
p3 = plot_with_shade(all_d, iter_FAFW, prc, alphaa, colors(3,:));
p4 = plot_with_shade(all_d, iter_MSK, prc, alphaa, colors(4,:));
set(gca, 'XScale', 'log', 'YScale', 'log');
set(gca, 'FontSize', font_size - 2);
xlabel('Dimension', 'FontSize', font_size);
ylabel('# iterations','FontSize', font_size)
grid on
lgd = legend([p1, p2, p3, p4], 'Vanilla FW', 'Adaptive FW', 'Fully Adaptive FW', 'MOSEK', 'Location', 'southeast');
lgd.FontSize = font_size;
ylim([4, 2e3]);
remove_border()
saveas(gcf, 'iteration', 'svg')