function sol = my_bisection(h, LB, UB, tol)
% A simple implementation of Bisection algorithm
%
% Syntax: sol = my_bisection(f, low, up, tol)
%
% Long description

    iter_max = ceil((log2(abs(UB-LB))-log2(tol)));
    h_lb = h(LB);
    for iter = 1 : iter_max
        sol = (LB + UB)/2;
        h_sol = h(sol);
        if h_lb * h_sol < 0 
            UB = sol;
        else
            LB = sol;
            h_lb = h_sol;
        end
    end
    sol = (LB + UB)/2;
    
end