function [x, y, flag, it, normr, resvec, errvec] = ...
    lnlq( A, b, atol, btol, etol, conlim, maxit, M, lambda, sigma )
% LNLQ Least-Norm LQ method
%   X = LNLQ(A,B) attempts to solve the system of linear equations
%   [I       A'   ][X] = [0]
%   [A -lambda^2 I][Y]   [b]
%   for X and Y. B is a column vector of length M. This is equivalent to
%     min   |X|^2
%     s.t. A*X = B
%   The system must be consistent.
%
%   [X,Y] = LNLQ(AFUN,B) accepts a function handle AFUN instead of the
%   matrix A. AFUN(X) accepts a vector input X and returns the
%   matrix-vector product A*X. In all of the following syntaxes, you can
%   replace A by AFUN.
%
%   [X,Y] = LNLQ(A,B,ATOL,BTOL) continues iterations until a certain
%   backward error estimate is smaller than some quantity depending on
%   ATOL and BTOL.  Let RES = B - A*X be the residual vector for the
%   current approximate solution X.  If A*X = B seems to be consistent,
%   LNLQ terminates when NORM(RES) <= ATOL*NORM(A)*NORM(X) + BTOL*NORM(B).
%
%   [X,Y] = LNLQ(A,B,ATOL,BTOL,ETOL,...,SIGMA) continues iterations until a
%   certain forward error estimate is satisfied. Let X* be the solution
%   then LNLQ will exit when NORM(X*-X) <= ETOL. SIGMA must be an
%   underestimate of the smallest singular value of A. If the exact minimum
%   singular value is known (call it S), then SIGMA should be S*(1-ETA)
%   where ETA is O(1e-10).
%
%   [X,Y] = LNLQ(A,B,ATOL,BTOL,ETOL,CONLIM) terminates if an estimate
%   of cond(A) exceeds CONLIM. For compatible systems Ax = b,
%   conlim could be as large as 1.0e+12 (say). If CONLIM is [], the default
%   value is CONLIM = 1e+8. Maximum precision can be obtained by setting 
%   ATOL = BTOL = CONLIM = 0, but the number of iterations may then be 
%   excessive.
%
%   [X,Y] = LNLQ(A,B,ATOL,BTOL,ETOL,MAXIT) specifies the maximum number of
%   iterations. If MAXIT is [] then SYMMLQ uses the default, min(N,20).
%
%   [X,Y] = LNLQ(A,B,ATOL,BTOL,ETOL,MAXIT,M) uses an SPD preconditioner M.
%   M should be supplied either as an explicit matrix or as a function such
%   that M(x) = M\x.
%
%   [X,Y] = LNLQ(A,B,...,LAMBDA) solves the regularized least squares
%   system
%     min   |X|^2 + |Z|^2
%     s.t. A*X + LAMBDA*Z = B
%   NOTE: Regularized problems must be run without preconditioning,
%         otherwise they will return the wrong answer.
%
%   [X,Y,FLAG] = LNLQ(A,B,...) also returns a convergence FLAG:
%    0 LNLQ converged to the desired tolerance
%      NORM(RES) <= ATOL*NORM(A)*NORM(X) + BTOL*NORM(B) within MAXIT
%      iterations.
%    1 LNLQ iterated MAXIT times but did not converge.
%    2 Estimation of COND(A) > CONLIM.
%    3 LNLQ converged to the least-squares solution
%      NORM(A^T*R)/(NORM(A)*NORM(R)) < ATOL within MAXIT iterations.
%    4 LNLQ converged to the desired 2-norm error tolerance
%      NORM(X*-X) <= ETOL within MAXIT iterations
%
%   [X,XCG,FLAG,ITER] = LNLQ(A,B,...) also returns the
%   iteration number at which X was computed: 0 <= ITER <= MAXIT.
%
%   [X,XCG,FLAG,ITER,NORMR] = LNLQ(A,B,...) also returns the relative
%   residual NORM(B-A*X)/NORM(B). If FLAG is 0, then
%   RELRES <= ATOL*NORM(A)*NORM(X) + BTOL*NORM(B).
%
%   [X,XCG,FLAG,ITER,NORMR,NORMAR,RESVEC] = LNLQ(A,B,...) also 
%   returns a vector of estimates of the LNLQ residual and optimality
%   residual norms at each iteration, including NORM(B-A*X0) and
%   NORM(A'*(B-A*X0)).
%
%   [X,XCG,FLAG,ITER,NORMR,NORMAR,RESVEC,RESVECAR,ERRVEC] = LNLQ(A,B,...)
%   also returns a vector of estimates of the LNLQ 2-norm error if
%   0 < SIGMA < MIN(SVD(A)).

% Initialization
  if isa(A,'numeric')
    explicitA = true;
  elseif isa(A,'function_handle')
    explicitA = false;
  else
    error('SOL:LNLQ:Atype','%s','A must be numeric or a function handle');
  end
  
  if nargin < 8  || isempty(M)
      M = @(x) x;
  end
  
  if isa(M,'numeric')
    explicitM = true;
  elseif isa(M,'function_handle')
    explicitM = false;
  else
    error('SOL:LNLQ:Mtype','%s','M must be numeric or a function handle');
  end
  
  % Determine dimensions m and n, and
  % form the first vectors u and v.
  % These satisfy  beta*u = b,  alpha*v = A'u.

  u    = b;
  if explicitM
    p = M\u;  
  else
    p = M(u);
  end
  beta = sqrt(u'*p);
  normb = beta;
  if beta > 0
    p  = p/beta;
  end

  if explicitA
    v = A'*p;
    [m, n] = size(A);
  else  
    v = A(p,2);
    m = size(b,1);
    n = size(v,1);
  end

  minDim = min([m n]);
  
  % Set default parameters.
  
  if nargin < 3  || isempty(atol)     , atol      = 1e-6;            end
  if nargin < 4  || isempty(btol)     , btol      = 1e-6;            end
  if nargin < 5  || isempty(etol)     , etol      = 1e-16;           end
  if nargin < 6  || isempty(conlim)   , conlim    = 1e+8;            end
  if nargin < 7  || isempty(maxit)    , maxit     = min(minDim, 20); end
  if nargin < 9  || isempty(lambda)   , lambda    = 0;               end
  if nargin < 10 || isempty(sigma)    , sigma     = 0;               end

  flag = 1;
  sigmax = 0;
  sigmin = Inf;
  reg = lambda;
  
  sigma = sqrt(lambda^2 + sigma^2);
  
  resvec = zeros(maxit+1,1);
  errvec = zeros(maxit+1,1);
  
  alpha = norm(v);
  if alpha > 0
    v = (1/alpha)*v;
  end
  
  % Modify Bk due to regularization
  if( lambda > 0 )
      alphaL = sqrt(alpha^2 + reg^2);
      cL = alpha/alphaL;
      sL = reg/alphaL;
      
      vL = cL*v;
      q = sL*v;
  else
      alphaL = alpha;
      vL = v;
  end
  
  anorm2 = alphaL;

  % Initial values
  rhobar = -sigma;
  csig = -1;
  it = 1;
  tau = normb / alphaL;
  x = tau * vL;
  yL = zeros(m,1);
  normx2 = tau * tau;
  normy2 = 0;
  
  % Initial LQ factorization
  epsbar = alphaL;
  
  % Initial values for yL
  zetabar = tau/epsbar;
  wbar = p;
  
  while 1

      % Golub-Kahan step
      if (explicitA)
        u = A * v  - (alpha/beta)*u;
      else
        u = A(v,1) - (alpha/beta)*u; 
      end
      if (explicitM)
          p = M\u;
      else
          p = M(u);
      end
      beta = sqrt(u'*p);
      if( beta > 0 )
          p = p/beta;
      end
      
      % Modify Bk due to regularization
      if( lambda > 0 )
          betaL = cL*beta;
          tempL = sL*beta;

          reg = sqrt(lambda^2 + tempL^2);
          cL = lambda/reg;
          sL = tempL/reg;
          q = sL*q;
      else
          betaL = beta;
      end
      
      % Continue QR factorization for error estimate
      if( sigma > 0 )
          mubar = -csig*alphaL;

          rho = sqrt(rhobar^2 + alphaL^2);
          csig = rhobar/rho;
          ssig = alphaL/rho;
          rhobar = ssig*mubar + csig*sigma;
          mubar = -csig*betaL;
          
          h = betaL*csig/rhobar;
          omega = sqrt(sigma^2 - sigma*betaL*h);
          
          rho = sqrt(rhobar^2 + betaL^2);
          csig = rhobar/rho;
          ssig = betaL/rho;
          rhobar = ssig*mubar + csig*sigma;
      end
      
      if (beta > 0)
        if (explicitA)
          v = A' * p - beta*v;
        else
          v = A(p,2) - beta*v;
        end
        alpha = norm(v);
        if (alpha > 0)
          v = v/alpha;
        end
      end

      % Modify Bk due to regularization
      if( lambda > 0 )
          alphaL = sqrt(alpha^2 + reg^2);
          cL = alpha/alphaL;
          sL = reg/alphaL;

          vL = cL*v + sL*q;
          q = sL*v - cL*q;
      else
          alphaL = alpha;
          vL = v;
      end
      
      % Update estimate of |A|_F
      anorm2 = anorm2 + betaL^2 + alphaL^2;
      
      % Continue error estimation
      if( sigma > 0 )
          tautilde = -tau * betaL / omega;
      end

      % Continue LQ factorization
      eps = sqrt(epsbar^2 + betaL^2);
      c = epsbar/eps;
      s = betaL/eps;
      eta = alphaL*s;
      epsbar = -alphaL*c;
      
      % Residual estimation in x
      normr = abs(betaL * tau);
      
      % Forward substitution of L for x
      tau = -tau * betaL / alphaL;
      x = x + tau * vL;
      normx2 = normx2 + tau*tau;
      
      % Forward substitution of M for yL
      zeta = c*zetabar;
      zetabar = (tau - eta*zeta)/epsbar;
      
      % Compute iterates
      w    = c*wbar + s*p;
      wbar = s*wbar - c*p;

      yL = yL - zeta*w;
      y  = yL - zetabar*wbar;
      normy2 = normy2 + zeta*zeta;
      
      % Check if should exit
      test0 = (normr <= atol*sqrt(anorm2) + btol*normb);
      test1 = (it > maxit);
      test4 = 0;
%      test2 = (sigmax/sigmin >= conlim);

      if( sigma > 0 )
          % Error estimate in x
          err_x = sqrt(tautilde^2 - tau^2);
          
          % Error estimate in y
          etatilde = omega*s;
          epstilde = -omega*c;
          zetatilde = (tautilde - etatilde*zeta)/epstilde;
          
          err_y = sqrt(zetatilde^2 - zetabar^2);
          
          err = sqrt(err_x^2 + err_y^2);

          errvec(it) = err;
          test4 = (err < etol);
      end

      if( test0 + test1 + test4 > 0 )
          resvec = resvec(1:it);
          errvec = errvec(1:it);
          
          % normy2 = normy2 + zetabar*zetabar;
          
          if test0, flag = 0; end
          if test1, flag = 1; end
%          if test2, flag = 2; end
          if test4, flag = 4; end
          
          break;
      end
      it = it+1;  
  end
end