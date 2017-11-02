function [ x, flag, iter, resvec ] = spqmrns( A, H1, H2, f, tol, maxiter, M )

% SPMR-SC solves saddle-point systems (for x only) of the form
% [A G1'][x] = [f]
% [G2 0 ][y] = [0]
% where A is an n-by-n matrix, and G1,G2 are m-by-n matrices.
% Recovering y needs to be done outside of the method.
%
% The input arguments to be passed in are:
%   A       : matrix or a function A(x,t) such that
%              A(x,1) = A *x
%              A(x,2) = A'*x
%   H1      : matrix or a function Z1(x,t) such that
%              Z1(x,1) = Z1 *x
%              Z1(x,2) = Z1'*x
%             where Z1 is a null-space basis for G1
%   H2      : matrix or a function Z2(x,t) such that
%              Z2(x,1) = Z2 *x
%              Z2(x,2) = Z2'*x
%             where Z2 is a null-space basis for G2
%   f       : an n-vector
%   tol     : the relative residual tolerance. Default is 1e-6.
%   maxiter : maximum number of iterations. Default is 20.
%   M       : a symmetric-positive definite preconditioner, accessible
%             as a function or matrix
%               M(x) = M\x
%
% The output variables are
%   x       : an n-vector
%   flag    : flag indicating result:
%              0 : residual norm is below tol
%              1 : ran maxit iterations but did not converge
%              2 : some computed quantity became too small
%   iter    : number of iterations
%   resvec  : a vector of length iter containing estimates of |rk|/|b|
%             where |rk| is the kth residual norm

if nargin < 4
    error('spqmrns:args','%s','Not enough input arguments');
end

if nargin < 5 || isempty(tol)      , tol     = 1e-6;       end
if nargin < 6 || isempty(maxiter)  , maxiter = 20;         end
if nargin < 7 || isempty(M) 
    precond = 0;       
else
    precond = 1;
end

if isa(A,'numeric')
    explicitA = true;
elseif isa(A,'function_handle')
    explicitA = false;
else
    error('spqmrns:Atype','%s','A must be numeric or a function handle');
end

if isa(H1,'numeric')
    explicitH1 = true;
elseif isa(H1,'function_handle')
    explicitH1 = false;
else
    error('spqmrns:H1type','%s','H1 must be numeric or a function handle');
end

if isa(H2,'numeric')
    explicitH2 = true;
elseif isa(H2,'function_handle')
    explicitH2 = false;
else
    error('spqmrns:H2type','%s','H2 must be numeric or a function handle');
end

if precond
    if isa(M,'numeric')
        explicitM = true;
    elseif isa(M,'function_handle')
        explicitM = false;
    else
        error('spmrns:Mtype','%s','M must be numeric or a function handle');
    end
end

flag = 1;
% tolerance before quantity considered too small (arbitrary for now)
eps = 1e-12;

n = length(f);
resvec = zeros(maxiter,1);
if explicitH1, Hf = -H1'*f; else Hf = -H1(f,2); end
m = length(Hf);
nf = norm(Hf);

z = Hf;
if precond
    if explicitM, Mz = M\z; else Mz = M(z); end
else
    Mz = z;
end
beta1 = sqrt(z'*Mz);
z = z/beta1;
v = z;
Mz = Mz/beta1;
if precond
    if explicitM, Mv = M\v; else Mv = M(v); end
else
    Mv = v;
end

if explicitH2, u = H2*Mv; else u = H2(Mv,1); end
if explicitH1, w = H1*Mz; else w = H1(Mz,1); end
if explicitA, Au = A*u; Aw = A'*w; else Au = A(u,1); Aw = A(w,2); end
alphgam = w'*Au;
if (abs(alphgam) < eps)
    x = 0;
    flag = 2;
    iter = 0;
    return;
end
Jold = sign(alphgam);
alpha = sqrt(abs(alphgam));
gamma = alpha;
u = Jold*u/alpha;
w = Jold*w/gamma;

beta = 0;
delta = 0;

p = zeros(n,1);

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
    if explicitH1
        vv = Jold*H1'*Au/alpha;
    else
        vv = Jold*H1(Au,2)/alpha;
    end
    v = vv - gamma*v;
    if precond
        if explicitM, Mv = M\v; else Mv = M(v); end
    else
        Mv = v;
    end
    if explicitH2
        zz = Jold*H2'*Aw/gamma;
    else
        zz = Jold*H2(Aw,2)/gamma;
    end
    z = zz - alpha*z;
    if precond
        if explicitM, Mz = M\z; else Mz = M(z); end
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
    if explicitH2
        u = H2*Mv/delta - Jold*beta*u;
    else
        u = H2(Mv,1)/delta - Jold*beta*u;
    end
    if explicitH1
        w = H1*Mz/beta - Jold*delta*w;
    else
        w = H1(Mz,1)/beta - Jold*delta*w;
    end
    if explicitA, Au = A*u; Aw = A'*w; else Au = A(u,1); Aw = A(w,2); end
    alphgam = w'*Au;
    if (abs(alphgam) < eps)
        flag = 2;
        break;
    end
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
        if explicitH1
            nr = norm(H1'*A*p - Hf);
        else
            nr = norm(H1(A*p,2) - Hf); 
        end
        if( nr < tol*nf )  
            flag = 0;
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