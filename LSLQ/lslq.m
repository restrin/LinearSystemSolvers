function [x, xcg, flag, it, normr, normAr, resvec, resvecAr, errvec, errveccg, errvecrcg] = ...
    lslq( A, b, atol, btol, etol, conlim, maxit, lambda, sigma, d )
% LSLQ Least-Squares LQ method
%   X = LSLQ(A,B) attempts to solve the system of linear equations A*X=B
%   for X. B is a column vector of length N. The system must be consistent.
%
%   X = LSLQ(AFUN,B) accepts a function handle AFUN instead of the matrix
%   A. AFUN(X) accepts a vector input X and returns the matrix-vector
%   product A*X. In all of the following syntaxes, you can replace A by
%   AFUN.
%
%   X = LSLQ(A,B,ATOL,BTOL) continues iterations until a certain
%   backward error estimate is smaller than some quantity depending on
%   ATOL and BTOL.  Let RES = B - A*X be the residual vector for the
%   current approximate solution X.  If A*X = B seems to be consistent,
%   LSLQ terminates when NORM(RES) <= ATOL*NORM(A)*NORM(X) + BTOL*NORM(B).
%   If the system is inconsistent, then we continue until
%   NORM(RES) <= ATOL * NORM(A) * NORM(RES).
%
%   X = LSLQ(A,B,ATOL,BTOL,ETOL,...,SIGMA,D) continues iterations until a
%   certain forward error estimate is satisfied. Let X* be the solution
%   then LSLQ will exit when NORM(X*-X) <= ETOL. SIGMA must be an
%   underestimate of the smallest singular value of A. If the exact minimum
%   singular value is known (call it S), then SIGMA should be S*(1-ETA)
%   where ETA is O(1e-10). D >= 0 is used to improve the error estimate,
%   where at iteration K we tighten the error bound at iteration K-D with
%   O(D) work.
%
%   X = LSLQ(A,B,ATOL,BTOL,ETOL,CONLIM) terminates if an estimate
%   of cond(A) exceeds CONLIM. For compatible systems Ax = b,
%   conlim could be as large as 1.0e+12 (say). If CONLIM is [], the default
%   value is CONLIM = 1e+8. Maximum precision can be obtained by setting 
%   ATOL = BTOL = CONLIM = 0, but the number of iterations may then be 
%   excessive.
%
%   X = LSLQ(A,B,ATOL,BTOL,ETOL,MAXIT) specifies the maximum number of
%   iterations. If MAXIT is [] then SYMMLQ uses the default, min(N,20).
%
%   X = LSLQ(A,B,...,LAMBDA) solves the regularized least squares system
%   min |[  A   ]*x - [b]|
%       |[LAMBDA]     [0]|_2.
%
%   [X,XCG] = LSLQ(A,B,...) also returns the LSQR transfer point.
%
%   [X,FLAG] = LSLQ(A,B,...) also returns a convergence FLAG:
%    0 LSLQ converged to the desired tolerance
%      NORM(RES) <= ATOL*NORM(A)*NORM(X) + BTOL*NORM(B) within MAXIT
%      iterations.
%    1 LSLQ iterated MAXIT times but did not converge.
%    2 Estimation of COND(A) > CONLIM.
%    3 LSLQ converged to the least-squares solution
%      NORM(A^T*R)/(NORM(A)*NORM(R)) < ATOL within MAXIT iterations.
%    4 LSLQ converged to the desired 2-norm error tolerance
%      NORM(X*-X) <= ETOL within MAXIT iterations
%
%   [X,XCG,FLAG,ITER] = LSLQ(A,B,...) also returns the
%   iteration number at which X was computed: 0 <= ITER <= MAXIT.
%
%   [X,XCG,FLAG,ITER,NORMR] = LSLQ(A,B,...) also returns the relative
%   residual NORM(B-A*X)/NORM(B). If FLAG is 0, then
%   RELRES <= ATOL*NORM(A)*NORM(X) + BTOL*NORM(B).
%
%   [X,XCG,FLAG,ITER,NORMR,NORMAR] = LSLQ(A,B,...) also returns A'*R, where
%   R = B - A*X. If FLAG is 0, then
%   NORMAR <= NORM(A)*ATOL*NORM(A)*NORM(X) + NORM(A)*BTOL*NORM(B).
%
%   [X,XCG,FLAG,ITER,NORMR,NORMAR,RESVEC,RESVECAR] = LSLQ(A,B,...) also 
%   returns a vector of estimates of the LSLQ residual and optimality
%   residual norms at each iteration, including NORM(B-A*X0) and
%   NORM(A'*(B-A*X0)).
%
%   [...,ERRVEC,ERRVECCG,ERRVECRCG] = LSLQ(A,B,...)
%   also returns a vector of estimates of the LSLQ 2-norm error if
%   0 < SIGMA < MIN(SVD(A)).

% Initialization
  if isa(A,'numeric')
    explicitA = true;
  elseif isa(A,'function_handle')
    explicitA = false;
  else
    error('SOL:lslq:Atype','%s','A must be numeric or a function handle');
  end
  
  flag = 1;
  sigmax = 0;
  sigmin = Inf;
  reg = lambda;
  % Determine dimensions m and n, and
  % form the first vectors u and v.
  % These satisfy  beta*u = b,  alpha*v = A'u.

  u    = b;
  beta = norm(u);
  n2b = beta;
  if beta > 0
    u  = u/beta;
  end

  if explicitA
    v = A'*u;
    [m, n] = size(A);
  else  
    v = A(u,2);
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
  if nargin < 8  || isempty(lambda)   , lambda    = 0;               end
  if nargin < 9  || isempty(sigma)    , sigma     = 0;               end
  if nargin < 10 || isempty(d)        , d         = 0;               end
  
  sigma = sqrt(lambda^2 + sigma^2);
  
  resvec = zeros(maxit,1);
  resvecAr = zeros(maxit,1);
  errvec = zeros(maxit,1);
  errveccg = zeros(maxit,1);
  errvecrcg = zeros(maxit,1);
  test4 = 0;
  
  alpha = norm(v);
  if alpha > 0
    v = (1/alpha)*v;
  end
  
  anorm2 = alpha;
  n2Ab = alpha*n2b;
  
  wbar = v;

  % Initial values
  rhobar = -sigma;
  gammabar = alpha;
  ss = n2b;
  c = -1;
  s = 0;
  delta = -1;
  tau = n2Ab;
  zeta = 0;
  it = 1;
  x = zeros(n,1);
  normx2 = 0;
  csig = -1;
  
  z_list = zeros(d,1);
  c_list = zeros(d,1);
  s_prod = ones(d,1);
  
  while 1

      % Golub-Kahan step
      if (explicitA)
        u = A * v  - alpha*u;
      else
        u = A(v,1) - alpha*u; 
      end
      beta = norm(u);

      if (beta > 0)
        u = u/beta;

        if (explicitA)
          v = A' * u - beta*v;
        else
          v = A(u,2) - beta*v;
        end
        alpha = norm(v);
        if (alpha > 0)
          v = v/alpha;
        end
      end
      
      % Modify Bk due to regularization
      if( lambda > 0 )
          betaL = sqrt(beta^2 + reg^2);
          cL = beta/betaL;
          sL = reg/betaL;
          alphaL = cL*alpha;
          tempL = sL*alpha;

          reg = sqrt(lambda^2 + tempL^2);
      else
          betaL = beta;
          alphaL = alpha;
      end

      % Update estimate of |A|_F
      anorm2 = anorm2 + betaL^2 + alphaL^2;
      
      % QR of Bk
      gamma = sqrt(gammabar^2 + betaL^2);
      
      % Forward sub for t
      tau = -tau*delta/gamma;

      % Continue QR of Bk
      cp = gammabar/gamma;
      sp = betaL/gamma;
      delta = sp*alphaL;
      gammabar = -cp*alphaL;

      % Continue QR factorization for error estimate
      if( sigma > 0 )          
          mubar = -csig*gamma;

          rho = sqrt(rhobar^2 + gamma^2);
          csig = rhobar/rho;
          ssig = gamma/rho;
          rhobar = ssig*mubar + csig*sigma;
          mubar = -csig*delta;
          
          h = delta*csig/rhobar;
          omega = sqrt(sigma^2 - sigma*delta*h);
          
          rho = sqrt(rhobar^2 + delta^2);
          csig = rhobar/rho;
          ssig = delta/rho;
          rhobar = ssig*mubar + csig*sigma;
      end
      
      % LQ of R
      epsbar = -gamma*c;
      eta = gamma*s;
      eps = sqrt(epsbar^2 + delta^2);
      c = epsbar/eps;
      s = delta/eps;
      
      % Condition number estimates
      sigmax = max([sigmax, gamma, gammabar]);
      sigmin = min([sigmin, gamma, gammabar]);
      
      % Residual estimate
      n2r = norm([ss*cp - zeta*eta; ss*sp]);
      resvec(it) = n2r;
      ss = ss*sp;
      
      % Forward sub for z, zbar
      zetaold = zeta;
      zeta = (tau - zeta*eta)/eps;
      zetabar = zeta/c;
      
      % Estimate A'r
      n2Ar = norm([gamma*eps*zeta; delta*eta*zetaold]);
      resvecAr(it) = n2Ar;
      
      % Transfer to CG point
      xcg = x + zetabar*wbar;
      normxcg = normx2 + zetabar^2;
      
      % Sliding window part of error estimate
      if d > 0 && sigma > 0          
          ix = mod(it-1,d)+1;
          
          if it > d+1
              jx = mod(ix-1,d)+1;
              zetabark = z_list(jx)/c_list(jx); 
              
              pix = mod(it-2,d)+1;
              theta = abs(c_list'*(s_prod.*z_list));
              theta = zetabark*theta ...
                  + abs(zetabark*zetabar*s_prod(pix)*s_old) ...
                  - zetabark^2;

              errveccg(it-d) = sqrt(errveccg(it-d)^2 - 2*theta);
          end
          
          c_list(ix) = c;
          z_list(ix) = zeta;  
          
          ix = mod(it-2,d)+1;
          if it < d && it > 1
             s_prod(it+1:end) = s_prod(it+1:end)*s_old; 
          elseif it > 1
             s_prod = s_prod/s_prod(mod(ix+1,d)+1);
             s_prod(mod(ix,d)+1) = s_prod(ix)*s_old; 
          end
          
          s_old = s;
      end
      
      if (it > 1 && sigma > 0)
          tau_tilde = (-tau*delta/omega);
          err_r = abs(tau_tilde);
          errvecrcg(it) = err_r;
          
          errvec(it) = abs(zetatilde);
          errveccg(it) = sqrt(zetatilde^2 - zetabar^2);
         
          test4 = (it > d+1) && (errveccg(it-d) <= etol);
      end
      
      % Check if should exit
      test0 = (n2r <= atol*sqrt(anorm2) + btol*n2b);
      test1 = (it > maxit);
      test2 = (sigmax/sigmin >= conlim);
      test3 = (n2Ar <= atol*sqrt(anorm2)*n2r);
      
      if( test0 + test1 + test2 + test3 + test4 > 0 )
          resvec = resvec(1:it);
          resvecAr = resvecAr(1:it);
          errvec = errvec(1:it);
          errveccg = errveccg(1:it);
          errvecrcg = errvecrcg(1:it);
          
          normr = n2r;
          normAr = n2Ar;
          
          if test0, flag = 0; end
          if test1, flag = 1; end
          if test2, flag = 2; end
          if test3, flag = 3; end
          if test4, flag = 4; end
          
          break;
      end

      % Compute iterates
      w    = c*wbar + s*v;
      wbar = s*wbar - c*v;

      x = x + zeta*w;
      normx2 = normx2 + zeta^2;

      % Continue process for error estimation
      if( sigma > 0 )        
          etatilde = omega*s;
          epstilde = -omega*c;
          tautilde = -tau*delta/omega;
          zetatilde = (tautilde - zeta*etatilde)/epstilde;
          errvec(it+1) = zetatilde;
      end
      
      it = it+1;
      
  end
  
end
