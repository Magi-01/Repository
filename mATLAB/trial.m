% x^2-2 = 0

function [root, ferr, convergence] = newton(x0, xtrue, fun, f1, tol, max_iter)
    % x_(n+1) = x_n - f(x_n)/f'(x_n)
    ferr = [];
    root = 0;
    err = [];
    convergence = [];
    conv = [];
    
    for i=1:max_iter
        xnext = x0 - fun(x0)/f1(x0);
        df = abs(xtrue - xnext);
        err = [err, df];
        if i > 1
            conv = [conv, err(i) / err(i-1)^2];
        end
        if df < tol
            ferr = [ferr, err];
            root = xnext;
            convergence = [convergence, conv];
            return
        end
        x0 = xnext;
    end
    error('The method did not converge');
end

clear;

x0 = pi;

fun = @(x) x^2 - 2;
f1 = @(x) 2*x;

xtrue = sqrt(2);

tol = 1e-6;

max_iter = 100;

format long

[root, ferr, convergence] = newton(x0, xtrue, fun, f1, tol, max_iter);

xfzero = abs(root - fzero(fun,x0));

disp('True value:');
disp(xtrue);

disp('True value found:');
disp(root);

disp('Difference between Newton method and fzero:');
disp(xfzero);

disp('Difference between Newton method and true value:');
disp(ferr(end));

disp('Convergence:');
disp(convergence);