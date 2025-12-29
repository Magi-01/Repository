function [vc, k, semilungh] = colebrook_bisection1(a, b)
% COLEBROOK_BISECTION Solve Colebrook–White equation using bisection
%
% [vc, k, semilungh] = colebrook_bisection(a, b)

    %% Default parameters
    E   = 1e-4;    % roughness
    d   = 1e-1;    % pipe diameter
    N   = 1e2;     % Reynolds number
    tol = 1e-11;
    maxit = ceil(log2((b-a)/tol));

    %% Colebrook–White function
    f = @(x) 1/sqrt(x) + 2*log10( (E/(3.7*d)) + (2.51./(N*sqrt(x))) );

    %% Initialize
    vc = [];
    semilungh = [];

    fa = f(a); fb = f(b);
    if fa*fb > 0
        error('Interval [%f, %f] does not bracket a root.', a, b);
    end

    %% Bisection loop
    for k = 1:maxit
        c = (a+b)/2;
        fc = f(c);

        semilun = (b-a)/2;

        % Store iteration data
        vc = [vc; c];
        semilungh = [semilungh; semilun];

        % Stop conditions
        if abs(fc) < tol || semilun < tol
            break
        end

        % Interval update
        if sign(fc) == sign(fa)
            a = c; fa = fc;
        else
            b = c; fb = fc;
        end
    end

    %% Ordered results (nice table)
    disp('Iteration results:')
    disp(table((1:k)', vc, f(vc), semilungh, ...
         'VariableNames', {'Iter','Midpoint','fMidpoint','HalfLength'}))

    %% Final check
    disp('Final approximation:')
    disp(vc(end))
    disp('f(x) at final approximation:')
    disp(f(vc(end)))

    %% Semi-logarithmic convergence plot
    figure;
    semilogy(1:k, semilungh, '-o','LineWidth',1.5);
    xlabel('Iteration');
    ylabel('Interval Half-Length (log scale)');
    title('Convergence of Colebrook–White Bisection Method');
    grid on;

end

%% Auto-run
if ~isdeployed
    format long
    [vc, k, semilungh] = colebrook_bisection1(0.08, 1);
end
