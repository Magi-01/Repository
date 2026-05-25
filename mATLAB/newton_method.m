function root_newton = newton_method3()
    % Initial guess
    x = 0.5;
    % Tolerance
    tol = 1e-6;
    % Maximum number of iterations
    max_iter = 100;
    for iter = 1:max_iter
        % Compute the function value
        fx = exp(-2*x) - x^2;
        % Compute the derivative
        dfx = -2*exp(-2*x) - 2*x;
        % Update the guess
        x_new = x - fx / dfx;
        % Check for convergence
        if abs(x_new - x) < tol
            root_newton = x_new;
            return;
        end
        x = x_new;
    end
    error('The method did not converge');
end

% Cambia il formato di visualizzazione a 'long' per vedere più cifre
format long

% Esegui il metodo di Newton
root_newton = newton_method3();

% Mostra il risultato del metodo di Newton
disp('Root found by Newton method:');
disp(root_newton);

% Definisci la funzione da usare con fzero
fun = @(x) exp(-2*x) - x^2;

% Usa fzero per trovare la radice
root_fzero = fzero(fun, 0.5);

% Mostra il risultato di fzero
disp('Root found by fzero:');
disp(root_fzero);

% Confronta i risultati
difference = abs(root_newton - root_fzero);
disp('Difference between Newton method and fzero:');
disp(difference);
