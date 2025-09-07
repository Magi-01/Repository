function [vc, k, steps] = newton2(F, J, x0, tol, maxit)
% NEWTON2 Solve F(x)=0 for a system using Newton's method
% Outputs:
%   vc    - iterates
%   k     - number of iterations
%   steps - ||x_{k+1}-x_k|| step norms

%% Initialize
vc = x0.';        % store iterates row-wise
steps = [];

%% Newton loop
for k = 1:maxit
    Fx  = F(x0);
    dFx = J(x0);
    
    x1  = x0 - dFx\Fx;   % Newton step
    
    step = norm(x1 - x0);   % step norm
    steps = [steps; step];
    vc = [vc; x1.'];
    errs = norm((1/2)*abs(dFx\Fx));
    
    % Check convergence
    if errs < tol
        break;
    end
    
    x0 = x1;
end

%% Diagnostics
Iter = (0:k)';                    % iteration index
X = vc(:,1);
Y = vc(:,2);
StepNorm = [0; steps];            % first step = 0
FNorm = zeros(k+1,1);             % ||F(x_k)||
StepRatio = nan(k+1,1);           % quadratic convergence ratio

for i = 1:k+1
    FNorm(i) = norm(F(vc(i,:)'));
    if i >= 3
        StepRatio(i) = StepNorm(i)/StepNorm(i-1)^2;
    end
end

% Display table
disp('Iteration results (step norm as error proxy):')
T = table(Iter, X, Y, StepNorm, FNorm, StepRatio, ...
    'VariableNames', {'Iter','x','y','StepNorm','FNorm','StepRatio'});
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
yF1_pos = sqrt(max(0, 2 - xgrid.^2));
yF1_neg = -yF1_pos;

% F2(x,y)=0: cube root real
yF2 = nthroot(2 - exp(xgrid-1), 3);

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
    F = @(x) [x(1)^2 + x(2)^2 - 2; exp(x(1)-1) + x(2)^3 - 2];
    J = @(x) [2*x(1) 2*x(2); exp(x(1)-1) 3*x(2)^2];
    x0 = [1.2; 1.2];
    tol = 1e-10;
    maxit = 50;
    [vc,k,steps] = newton2(F,J,x0,tol,maxit);
end
