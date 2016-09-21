function [ x, iter, resvec ] = spqmrns( A, Z1, Z2, f, tol, maxiter, M )

% SPMR-SC solves saddle-point systems (for x only) of the form
% [A G1'][x] = [f]
% [G2 0 ][y] = [0]
% where A is an n-by-n matrix, and G1,G2 are m-by-n matrices.
% Recovering y needs to be done outside of the method.
%
% The input arguments to be passed in are:
%   A       : a function A(x,t) such that
%              A(x,1) = A *x
%              A(x,2) = A'*x
%   Z1      : a function Z1(x,t) such that
%              Z1(x,1) = Z1 *x
%              Z1(x,2) = Z1'*x
%             where Z1 is a null-space basis for G1
%   Z2      : a function Z2(x,t) such that
%              Z2(x,1) = Z2 *x
%              Z2(x,2) = Z2'*x
%             where Z2 is a null-space basis for G2
%   f       : an n-vector
%   tol     : the relative residual tolerance. Default is 1e-6.
%   maxiter : maximum number of iterations. Default is 20.
%   M       : a symmetric-positive definite preconditioner, accessible
%             as a function
%               M(x) = M\x
%
% The output variables are
%   x       : an n-vector
%   y       : an m-vector
%   iter    : number of iterations
%   resvec  : a vector of length iter containing estimates of |rk|/|b|
%             where |rk| is the kth residual norm


if nargin < 5 || isempty(tol)      , tol     = 1e-6;       end
if nargin < 6 || isempty(maxiter)  , maxiter = 20;         end
if nargin < 7 || isempty(M) 
    precond = 0;       
else
    precond = 1;
end

n = length(f);
resvec = zeros(maxiter,1);
Zf = -Z1(f,2);
m = length(Zf);
nf = norm(Zf);

z = Zf;
if precond
    Mz = M(z);
else
    Mz = z;
end
beta1 = sqrt(z'*Mz);
z = z/beta1;
v = z;
Mz = Mz/beta1;
if precond
    Mv = M(v);
else
    Mv = v;
end

u = Z2(Mv,1);
w = Z1(Mz,1);
Au = A*u;
Aw = A'*w;
alphgam = w'*Au;
Jold = sign(alphgam);
alpha = sqrt(abs(alphgam));
gamma = alpha;
u = Jold*u/alpha;
w = Jold*w/gamma;

beta = 0;
delta = 0;

p = zeros(m,1);

% QR factorization of C_k
rhobar = gamma;

% Solving for x
phiold = beta1;
ww = u;

% Residual estimation
normr = 1; % ||r||/||b||

% Iteration count
iter = maxiter;

for k = 1:maxiter
    % Get next v and z
    vv = Jold*Z1(Au,2)/alpha;
    v = vv - gamma*v;
    if precond
        Mv = M(v);
    else
        Mv = v;
    end
    zz = Jold*Z2(Aw,2)/gamma;
    z = zz - alpha*z;
    if precond
        Mz = M(z);
    else
        Mz = z;
    end
    betdel = z'*Mv;
    delta = sqrt(abs(betdel));
    beta = betdel/delta;
    z = z/beta;
    v = v/delta;
    %============

    % Get next u and w
    u = Z2(Mv,1)/delta - Jold*beta*u;
    w = Z1(Mz,1)/beta - Jold*delta*w;
    Au = A*u;
    Aw = A'*w;
    alphgam = w'*Au;
    J = sign(alphgam);
    alpha = sqrt(abs(alphgam));
    gamma = alpha;
    u = J*u/alpha;
    w = J*w/gamma;
    %============

    % Update QR factorization of C_k
    rho = norm([rhobar delta]);
    c = rhobar/rho;
    s = delta/rho;

    rhobar = -c*gamma;
    sigma = s*gamma;
    %============
    
    % Solve for p
    phi = s*phiold;
    phiold = c*phiold;
    
    p = p + (phiold/rho)*ww;
    ww = u - (sigma/rho)*ww;
    %============
    
    % Residual estimation
    normr = normr*s;
    resvec(k) = normr*sqrt(k);
    if (normr*sqrt(k) < tol)
        if( norm(Z1(A*p,2) - Zf) < tol*nf )    
            iter = k;
            break;
        end
    end
    %============
    
    % Variable reset
    phiold = phi;
    Jold = J;
    %===============
end

x = -p;
resvec = resvec(1:iter);

end