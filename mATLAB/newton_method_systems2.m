function [vc, k, steps] = newton2(F, J, x0, tol, maxit)
% NEWTON2 Solve F(x)=0 for a system using Newton's method
% Computes step norms, F norms, and post-iteration true error & quadratic ratio

%% Initialize
vc = x0.';        
steps = [];

%% Newton loop
for k = 1:maxit
    Fx  = F(x0);
    dFx = J(x0);
    
    % Newton update
    stepvec = dFx\Fx;
    x1 = x0 - stepvec;
    
    % Step size
    step = norm(x1 - x0);
    steps = [steps; step];
    vc = [vc; x1.'];
    
    % Stopping condition
    if step < tol
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

% Use last iterate as "true root" to compute true error and ratio
x_true = vc(end,:)';
ErrorNorm = arrayfun(@(i) norm(vc(i,:)' - x_true), 1:k+1)';
ErrorRatio = [NaN; ErrorNorm(2:end)./ErrorNorm(1:end-1).^2];

% Display table
disp('Iteration results:')
T = table(Iter, X, Y, StepNorm, FNorm, ErrorNorm, ErrorRatio, ...
    'VariableNames', {'Iter','x','y','StepNorm','FNorm','ErrorNorm','ErrorRatio'});
disp(T)

% Final approximation
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

% F2(x,y)=0: x2 = x1^2 + x1
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

%% Example auto-run
if ~isdeployed
    format long
    F = @(x) [x(1)^2 + x(2)^2 - 1; x(1)^2 + x(1) - x(2)];
    J = @(x) [2*x(1) 2*x(2); 2*x(1)+1 -1];
    x0 = [1; 1];
    tol = 1e-10;
    maxit = 50;
    [vc,k,steps] = newton2(F,J,x0,tol,maxit);
end
