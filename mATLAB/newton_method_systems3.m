function [vc, k, steps] = newton2(F, J, x0, x_true, tol, maxit)
% NEWTON2 Solve F(x)=0 for a system using Newton’s method
% Tracks true error norms and quadratic convergence ratio.

%% Initialize
vc = x0.';        
steps = [];

ErrorNorm = norm(x0 - x_true);   % true error
ErrorRatio = NaN;                % undefined at first

%% Newton loop
for k = 1:maxit
    Fx  = F(x0);
    dFx = J(x0);
    
    % Newton update
    stepvec = dFx\Fx;
    x1 = x0 - stepvec;
    
    % track true error
    ErrorNorm = [ErrorNorm; norm(x1 - x_true)];
    
    % quadratic convergence ratio
    if k >= 1
        ratio = ErrorNorm(end)/ErrorNorm(end-1)^2;
    else
        ratio = NaN;
    end
    ErrorRatio = [ErrorRatio; ratio];
    
    % step size
    step = norm(x1 - x0);
    steps = [steps; step];
    vc = [vc; x1.'];
    
    % stopping condition
    if norm(x1 - x0) < tol
        break;
    end
    
    x0 = x1;
end

%% Diagnostics
Iter = (0:k)';                    
X = vc(:,1);
Y = vc(:,2);
StepNorm = [0; steps];           
FNorm = arrayfun(@(i) norm(F(vc(i,:)')), 1:k+1)';

disp('Iteration results:')
T = table(Iter, X, Y, StepNorm, FNorm, ErrorNorm, ErrorRatio, ...
    'VariableNames', {'Iter','x','y','StepNorm','FNorm','ErrorNorm','ErrorRatio'});
disp(T)

disp('Final approximation:')
disp(vc(end,:))
disp('F(x) at final approximation:')
disp(F(vc(end,:)'))

%% Plot curves and Newton iterates
figure; hold on; grid on;
xgrid = linspace(min(vc(:,1))-0.5, max(vc(:,1))+0.5, 400);

% F1(x,y)=0: circle
yF1_pos = sqrt(max(0, 1 - xgrid.^2));
yF1_neg = -yF1_pos;

% F2(x,y)=0: parabola
yF2 = xgrid.^2 + xgrid;

plot(xgrid, yF1_pos,'b','LineWidth',1.5);
plot(xgrid, yF1_neg,'b','LineWidth',1.5);
plot(xgrid, yF2,'r','LineWidth',1.5);

% Newton iterates
plot(vc(:,1), vc(:,2), 'ko-', 'MarkerFaceColor','g','MarkerSize',6,'LineWidth',1.2);

xlabel('x'); ylabel('y');
title('Newton Iterates and System Curves');
legend('F1=0','F1=0','F2=0','Newton iterates','Location','best');
hold off;

end

%% Auto-run
if ~isdeployed
    format long
    F = @(x) [x(1)^2 + x(2)^2 - 1; x(1)^2 + x(1) - x(2)];
    J = @(x) [2*x(1) 2*x(2); 2*x(1)+1 -1];
    x0 = [-1; 1];
    x_true = [-1; 0];   % actual root
    tol = 1e-12;
    maxit = 50;
    [vc,k,steps] = newton2(F,J,x0,x_true,tol,maxit);
end
