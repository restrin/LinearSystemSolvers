%% Some example uses of SPMR-NS and SPQMR-NS

%% Example: Maxwells Equation
%% Set up the problem
load('matrices_homogeneous3.mat')

G1 = B;
G2 = B;
K = [A G1';G2 sparse(m,m)];
f = rand(n,1);
tol = 1e-10;
maxiter = 20;

%% Using -NS family of solvers
%% First example: using an SPD preconditioner
%% Create function handle for projections
H1 = @(x,t) (builtin('_paren', [eye(n) G1'; G1 zeros(m)]\[x; zeros(m,1)], 1:n));
H2 = @(x,t) (builtin('_paren', [eye(n) G2'; G2 zeros(m)]\[x; zeros(m,1)], 1:n));
P = @(x,t) (builtin('_paren', [A+M G1'; G2 zeros(m)]\[x;zeros(m,1)], 1:n));

%% Run various solvers

[ x, flag, iter, resvec ] = ...
    spmrns( A, H1, H2, f, tol, maxiter, P );

[ x2, flag2, iter2, resvec2 ] = ...
    spqmrns( A, H1, H2, f, tol, maxiter, P ); % Remember resvec is a strict upper bound

%% Plot results
figure;
semilogy(1:length(resvec), resvec, '-', 'LineWidth', 2);
hold all
semilogy(1:length(resvec2), resvec2, '-', 'LineWidth', 2);
title('Residual');
xlabel('Iteration');
legend('SPMR-NS','SPQMR-NS');

%% Second example: using general preconditioners
% Note: Maxwell is a symmetric problem so one wouldn't use this
%       preconditioning style, but it may be necessary for nonsymmetric
%       problems. It's included so that the code can be copied and pasted
%       elsewhere.
%% Create function handles for projections
P  = A+M;
H1 = @(x,t) (builtin('_paren', [eye(n) G1'; G1 zeros(m)]\[x; zeros(m,1)], 1:n));
H2 = @(x,t) ((t==1)*(builtin('_paren', [P G1'; G2 zeros(m)]\[x; zeros(m,1)], 1:n)) ...
           + (t==2)*(builtin('_paren', [P' G2'; G1 zeros(m)]\[x; zeros(m,1)], 1:n)));

%% Run various solvers

[ x, flag, iter, resvec ] = ...
    spmrns( A, H1, H2, f, tol, maxiter, [] );

[ x2, flag2, iter2, resvec2 ] = ...
    spqmrns( A, H1, H2, f, tol, maxiter, [] ); % Remember resvec is a strict upper bound

%% Plot results
figure;
semilogy(1:length(resvec), resvec, '-', 'LineWidth', 2);
hold all
semilogy(1:length(resvec2), resvec2, '-', 'LineWidth', 2);
title('Residual');
xlabel('Iteration');
legend('SPMR-NS','SPQMR-NS');