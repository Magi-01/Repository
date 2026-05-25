function [vc, k, steps] = colebrook_newton_1(x0, maxit)
% COLEBROOK_NEWTON_SAFE  Solve Colebrook–White equation using Newton’s method
%
% Outputs:
%   vc    - vector of iterates
%   k     - number of iterations performed
%   steps - vector of step sizes

    %% Parameters
    E = 1e-4;   % roughness
    d = 1e-1;   % diameter
    N = 1e2;    % Reynolds number
    tol = 1e-7;

    %% Colebrook–White in log10 form
    f  = @(x) 1/sqrt(x) + 2*log10( E/(3.7*d) + 2.51./(N*sqrt(x)) );

    %% Derivative of f(x)
    f1 = @(x) -(1./(2*x.^(3/2))) ...
              - (2.51 ./ (N*2*x.^(3/2))) ./ ( (E/(3.7*d)) + (2.51./(N*sqrt(x))) );

    %% Initialize
    vc = [];
    steps = [];

    %% Newton loop
    for k = 1:maxit
        fx  = f(x0);
        dfx = f1(x0);
        x1  = x0 - fx/dfx;

        % Store iteration history
        vc    = [vc; x1];
        steps = [steps; abs(x1 - x0)];

        % Check stop condition
        if abs(fx) < tol || abs(x1 - x0) < tol
            break;
        end
        x0 = x1;
    end

    %% Display iteration results
    disp('Iteration results:')
    disp(table((1:k)', vc, steps, 'VariableNames', {'Iter','x','Step'}))

    %% Verify last approximation
    disp('Final approximation:')
    disp(vc(end))
    disp('f(x) at final approximation:')
    disp(f(vc(end)))

    %% Semi-logarithmic convergence plot
    figure;
    semilogy(1:k, steps, '-o','LineWidth',1.5);
    xlabel('Iteration');
    ylabel('Step size (log scale)');
    title('Convergence of Colebrook–White Newton Method');
    grid on;

end

%% Auto-run
if ~isdeployed
    format long
    start = 0.08;
    it = 50;
    [vc, k, steps] = colebrook_newton_1(start, it);
end
