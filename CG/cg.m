function [x, flag, iter, normr, resvec, errvec] = ...
    cg( A, b, atol, btol, etol, maxit, M, safe, eigest, d )
% CG (Hestenes-Stiefel) Conjugate Gradient Method
%   X = CG(A,B) attempts to solve the system of linear equations A*X=B
%   for X. B is a column vector of length N. The system should be symmetric
%   positive semidefinite and consistent.
%
%   X = CG(AFUN,B) accepts a function handle AFUN instead of the matrix
%   A. AFUN(X) accepts a vector input X and returns the matrix-vector
%   product A*X. In all of the following syntaxes, you can replace A by
%   AFUN.
%
%   X = CG(A,B,ATOL,BTOL) continues iterations until a certain
%   backward error estimate is smaller than some quantity depending on
%   ATOL and BTOL.  Let RES = B - A*X be the residual vector for the
%   current approximate solution X.  If A*X = B seems to be consistent,
%   CG terminates when NORM(RES) <= ATOL*NORM(A)*NORM(X) + BTOL*NORM(B).
%   If the system is inconsistent, then we continue until
%   NORM(RES) <= ATOL * NORM(A) * NORM(RES).
%
%   X = CG(A,B,ATOL,BTOL,ETOL,...,EIGEST,D) continues iterations until a
%   certain forward error estimate is satisfied. Let X* be the solution
%   then CG will exit when NORM(X*-X) <= ETOL. EIGEST must be an
%   underestimate of the smallest eugenvalue of A. If the exact minimum
%   singular value is known (call it S), then EIGEST should be S*(1-ETA)
%   where ETA is O(1e-10). D >= 0 is used to improve the error estimate,
%   where at iteration K we tighten the error bound at iteration K-D with
%   O(D) work.
%
%   X = CG(A,B,ATOL,BTOL,ETOL,MAXIT) specifies the maximum number of
%   iterations.
%   
%   X = CG(A,B,ATOL,BTOL,ETOL,MAXIT,M) will run preconditioned CG if M
%   is a positive-definite preconditioner. It may be a function handle
%   such that M(x) = M\x.
%
%   X = CG(A,B,ATOL,BTOL,ETOL,MAXIT,M,SAFE) with SAFE=1 will run CG and
%   return an error if A or M are not positive definite if some value
%   becomes negative or zero. If SAFE=0, then CG will not throw errors if
%   A or M are not SPD (but it may not converge!).
%
%   [X,FLAG] = CG(A,B,...) also returns a convergence FLAG:
%    0 CG converged to the desired tolerance
%      NORM(RES) <= ATOL*NORM(A)*NORM(X) + BTOL*NORM(B) within MAXIT
%      iterations.
%    1 CG iterated MAXIT times but did not converge.
%    4 CG converged to the desired 2-norm error tolerance
%      NORM(X*-X) <= ETOL within MAXIT iterations
%    5 Computed value is either too small, too large, or negative when
%      it should be positive (applies only if SAFE>0).
%
%   [X,FLAG,ITER] = CG(A,B,...) also returns the
%   iteration number at which X was computed: 0 <= ITER <= MAXIT.
%
%   [X,FLAG,ITER,NORMR] = CG(A,B,...) also returns the relative
%   residual NORM(B-A*X)/NORM(B). If FLAG is 0, then
%   RELRES <= ATOL*NORM(A)*NORM(X) + BTOL*NORM(B).
%
%   [X,FLAG,ITER,NORMR,NORMAR,RESVEC] = CG(A,B,...) also 
%   returns a vector of estimates of the CG residual at each iteration,
%   NORM(B-A*X0).
%
%   [...,ERRVEC] = CG(A,B,...)
%   also returns a vector of estimates of the CG 2-norm error if
%   0 < EIGEST < MIN(EIG(A)).

% Initialization
  n = length(b);

  if isa(A,'numeric')
    explicitA = true;
  elseif isa(A,'function_handle')
    explicitA = false;
  else
    error('SOL:CG:Atype','%s','A must be numeric or a function handle');
  end
  
  if nargin < 7  || isempty(M)
      M = @(x) x;
  end
  
  if isa(M,'numeric')
    explicitM = true;
  elseif isa(M,'function_handle')
    explicitM = false;
  else
    error('SOL:LNLQ:Mtype','%s','M must be numeric or a function handle');
  end
  
  flag = 1;
  normb = norm(b);
  normA = 0;
  % Set default parameters.
  
  if nargin < 3  || isempty(atol)     , atol      = 1e-6;            end
  if nargin < 4  || isempty(btol)     , btol      = 1e-6;            end
  if nargin < 5  || isempty(etol)     , etol      = 1e-16;           end
  if nargin < 6  || isempty(maxit)    , maxit     = min(n, 20);      end
  if nargin < 8  || isempty(safe)     , safe      = 1;               end
  if nargin < 9  || isempty(eigest)   , eigest    = 0;               end
  if nargin < 10 || isempty(d)        , d         = 0;               end
  
  x = zeros(n,1);
  
  if normb < btol
      flag = 0;
      iter = 0;
      normr = normb;
      resvec = normb;
      if eigest > 0
        errvec = normb / eigest;
      else
        errvec = 0;
      end
      
      return;
  end
  
  resvec = zeros(maxit,1);
  errvec = zeros(maxit,1);
  
  if explicitA
      r = b - A * x;
  else
      r = b - A(x);
  end
  
  if explicitM
      z = M\r;
  else
      z = M(r);
  end
  
  p = z;
  rzold = r' * z;

  gammacgold = -1;
  deltacgold = 0;
  s = 0;
  c = -1;
  epsilonold = 0;
  zeta = 0;
  zetaold = 0;
  
  for it = 1:maxit
      if explicitA
        Ap = A * p;
      else
        Ap = A(p);
      end
      pAp = p' * Ap;
      if safe && pAp <= 0
          flag = 5;
          break;
      end
      
      gammacg = rzold / pAp;
      x = x + gammacg * p;
      r = r - gammacg * Ap;
      
      rtol = atol*sqrt(normA) + btol*normb;
      normr = norm(r);
      resvec(it) = normr;
      test0 = normr < rtol;
      
      if test0
          flag = 0;
          break;
      end
      
      if explicitM
          z = M\r;
      else
          z = M(r);
      end      

      rz = r' * z;
      if safe && rz <= 0
          flag = 5;
          break;
      end
      
      deltacg = rz / rzold;        
      p = z + deltacg * p;
      rzold = rz;
        
      alpha = 1/gammacg + deltacgold/gammacgold;
      beta = sqrt(deltacg)/gammacg;
      normA = normA + alpha^2 + beta^2;
      
      % Error estimation below
      if eigest > 0
      if it == 1
          gammabar = alpha;
          deltabar = beta;
          
          % QR of T
          gamma = sqrt(gammabar^2 + beta^2);
          c = gammabar/gamma;
          s = beta/gamma;
          
          % Shifted QR
          rhobar = alpha - eigest;
          sigmabar = beta;
            
          omega = eigest + beta^2/rhobar;
            
          rho = sqrt(rhobar^2 + beta^2);
          cw = rhobar/rho;
          sw = beta/rho;
            
          zetabar = normb/gammabar;
          
          % TODO: double-check that this is okay
          errvec(1) = sqrt(normb^2/eigest^2 - zetabar^2);
      else
          % QR of T
          gammabar = s*deltabar - c*alpha;
          delta = c*deltabar + s*alpha;

          % Forward substitution
          zeta = zetabar*c;
          zetabar = -(delta*zeta + epsilonold*zetaold)/gammabar;

          % Forward substitution for error
          psi = c*deltabar + s*omega;
          omegabar = s*deltabar - c*omega;
          zetatilde = -(epsilonold*zetaold + psi*zeta)/omegabar;
                        
          errvec(it) = sqrt(zetatilde^2 - zetabar^2);
          if errvec(it) < etol
              flag = 4;
              break;
          end
          
          % Continue QR
          epsilon = s*beta;
          deltabar = -c*beta;
          
          gamma = sqrt(gammabar^2 + beta^2);
          c = gammabar/gamma;
          s = beta/gamma;
          
          % Shifted QR
          rhobar = sw*sigmabar - cw*(alpha - eigest);
          sigmabar = -cw*beta;
            
          omega = eigest - beta^2*cw/rhobar;
            
          rho = sqrt(rhobar^2 + beta^2);
          cw = rhobar/rho;
          sw = beta/rho;
            
          % Replacement
          epsilonold = epsilon;
          zetaold = zeta;
      end
      end
      
      deltacgold = deltacg;
      gammacgold = gammacg;
  end

  iter = it;
  normr = normb;
  resvec = resvec(1:iter);
  if flag == 0
    % If exit due to small residual, we quit before computing the next
    % error bound to avoid applying the preconditioner an extra time.
    errvec = errvec(1:iter-1);
  else
    errvec = errvec(1:iter);
  end
  
end