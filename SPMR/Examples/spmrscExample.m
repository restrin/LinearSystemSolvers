%% Some example uses of SPMR-SC and SPQMR-SC

%% Example: Highly non-normal matrices
%% Set up the problem
n = 1000;
m = 500;

A = gallery('grcar', n);
F = sprand(m,n/2,0.1) + 1e2*eye(m);
G1 = [F F];
G2 = G1;

K = [A G1';G2 zeros(m)];
g = rand(m,1);
x_exact = K\[zeros(n,1);g];
tol = 1e-10;
maxiter = 2*m;

%% Using -SC family of solvers
%% Create function handle for A-solves
% Not the most efficient way to do it, but serves as demonstration
[L,U,P] = lu(A);
Afun = @(x,t) ((t==1)*(U\(L\(P'*x))) + (t==2)*(P*(L'\(U'\x))));

%% Run various solvers

[ x, y, flag, iter, resvec ] = ...
    spmrsc( Afun, G1, G2, g, tol, maxiter, speye(m) );

[ x2, y2, flag2, iter2, resvec2 ] = ...
    spqmrsc( Afun, G1, G2, g, tol, maxiter, speye(m) ); % Remember resvec is a strict upper bound

%% Plot results
figure;
semilogy(1:length(resvec), resvec, '-', 'LineWidth', 2);
hold all
semilogy(1:length(resvec2), resvec2, '-', 'LineWidth', 2);
title('Residual');
xlabel('Iteration');
legend('SPMR-SC','SPQMR-SC');