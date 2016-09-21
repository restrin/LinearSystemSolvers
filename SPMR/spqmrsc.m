function [ x, y, iter, resvec ] = spqmrsc( A, G1, G2, g, tol, maxiter, M )

% SPQMR-SC solves saddle-point systems of the form
% [A G1'][x] = [0]
% [G2 0 ][y] = [g]
% where A is a nonsingular n-by-n matrix, and G1,G2 are m-by-n matrices.
%
% The input arguments to be passed in are:
%   A       : a function A(x,t) such that
%              A(x,1) = A \x
%              A(x,2) = A'\x
%   G1      : a function G1(x,t) such that
%              G1(x,1) = G1 *x
%              G1(x,2) = G1'*x
%   G2      : a function G2(x,t) such that
%              G2(x,1) = G2 *x
%              G2(x,2) = G2'*x
%   g       : an m-vector
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

m = length(g);
ng = norm(g);

resvec = zeros(maxiter,1);

% First iteration
z = g;
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

Gv = G1(Mv,2);
u = A(Gv,1);
Gz = G2(Mz,2);
w = A(Gz,2);
alphgam = w'*Gv;
Jold = sign(alphgam);
alpha = Jold*sqrt(abs(alphgam));
gamma = alpha;
u = Jold*u/alpha;
w = Jold*w/gamma;

n = length(u);

beta = 0;
delta = 0;

x = zeros(n,1);
y = zeros(m,1);

% QR factorization of C_k
rhobar = gamma;

% Solving for x
phiold = beta1;
ww = u;

% Solving for y
alphaold = alpha;
sigmaold = 0;
P = zeros(m,3);
P(:,3) = v;
mu = 0;
nu = 0;

% Residual estimation
normr = 1; % ||r||/||b||

% Iteration count
iter = maxiter;

for k = 1:maxiter
    % Get next v and z
    v = G2(u,1) - gamma*v;
    if precond
        Mv = M(v);
    else
        Mv = v;
    end
    z = G1(w,1) - alpha*z;
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
    Gv = (G1(Mv,2))/delta;
    u = A(Gv,1) - Jold*beta*u;
    Gz = (G2(Mz,2))/beta;
    w = A(Gz,2) - Jold*delta*w;
    alphgam = w'*Gv;
    J = sign(alphgam);
    alpha = J*sqrt(abs(alphgam));
    gamma = alpha;
    u = J*u/alpha;
    w = J*w/gamma;
    %============
    
    % Update QR factorization of Ck
    rho = norm([rhobar delta]);
    c = rhobar/rho;
    s = delta/rho;

    rhobar = -c*gamma;
    sigma = s*gamma;
    %============
    
    % Solve for x
    phi = s*phiold;
    phiold = c*phiold;
    
    x = x + (phiold/rho)*ww;
    ww = u - (sigma/rho)*ww;
    %============
    
    % Solve for y
    lambda = Jold*alphaold*rho;
    
    P(:,3) = (P(:,3) - mu*P(:,2) - nu*P(:,1))/lambda;
    y = y - phiold*P(:,3);
    
    P(:,1) = P(:,2); P(:,2) = P(:,3); P(:,3) = v;
    mu = Jold*beta*rho + J*alpha*sigma;
    nu = Jold*beta*sigmaold;
    %============
    
    % Residual estimation
    normr = normr*s;
    resvec(k) = normr*sqrt(k);
    if (normr*sqrt(k) < tol)
        if( norm(G2(x,1) - g) < tol*ng )    
            iter = k;
            break;
        end
    end
    %============
    
    % Variable reset
    phiold = phi;
    alphaold = alpha;
    sigmaold = sigma;
    Jold = J;
    %===============
end

% Recover y
if precond == 1, y = M(y); end
resvec = resvec(1:iter);

end