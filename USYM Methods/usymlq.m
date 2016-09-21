function [x, istop, itn, rnorm, Anorm, xnorm] = ...
    usymlq(A,b,c,atol,btol,maxit)

    %------------------------------------------------------------------
    % USYMLQ finds a solution x to the system of linear equations Ax = b
    % where A is a real nonsingular matrix with m rows and n columns,
    % and b is a real m-vector.
    % The matrix A is treated either as a matrix or a linear operator
    % accessed via functions:
    %   A(x,1) := A *x
    %   A(x,2) := A'*x
    %
    % USYMLQ uses an iterative method to approximate the solution.
    % The number of iterations required to reach a certain accuracy
    % depends strongly on the scaling of the problem.  Poor scaling of
    % the rows or columns of A should therefore be avoided where possible.
    % In the absence of better information, the nonzero columns of A
    % should be scaled so that they all have the same Euclidean norm.
    %
    % istop   Output     An integer giving the reason for termination...
    %
    %            0       x = 0 is the exact solution.
    %                    No iterations were performed.
    %
    %            1       norm(Ax - b) is sufficiently small,
    %                    given the values of atol and btol.
    %
    %            4       norm(Ax - b) is as small as seems reasonable.
    %
    %            7       The iteration limit itnlim was reached.
    %-------------------------------------------------------------------


    if isa(A,'numeric')
        explicitA = true;
    elseif isa(A,'function_handle')
        explicitA = false;
    else
        error('SOL:usymlq:Atype','%s','A must be numeric or a function handle');
    end

    m = length(b);
    n = length(c);
    
    if nargin < 4 || isempty(atol)      , atol    = 1e-6;       end
    if nargin < 5 || isempty(btol)      , btol    = 1e-6;       end
    if nargin < 6 || isempty(maxit)     , maxit   = min([m n]); end

    % Initialize
    xxnorm = 0;
    itn    = 0;
    istop  = 0;
    
    x = zeros(n,1);
    
    % Set up the first vectors for the tridiagonalization
    
    beta1  = norm(b);
    p     = b/beta1;
    gama1 = norm(c);
    q     = c/gama1;
    
    w = q;
    if explicitA
        u = A*q;
        v = A'*p;
    else
        u = A(q,1);
        v = A(p,2);
    end
    alfa = p'*u;
    u = u - alfa*p;
    v = v - alfa*q;
    beta = norm(u);
    u = u/beta;
    gama = norm(v);
    v = v/gama;
    
    AAnorm = alfa^2 + gama^2;
    gbar   = alfa;
    dbar   = beta;
    bnorm  = beta1;
    qrnorm = beta1;
    rhs1   = beta1;
    rhs2   = 0;
    
    for itn=1:maxit
        % Perform next step of tridiagonalization
        oldg = gama;
        if explicitA
            p = A*v - gama*p;
            q = A'*u - beta*q;
        else
            p = A(v,1) - gama*p;
            q = A(u,2) - beta*q;
        end
        alfa = p'*u;
        
        t = u;
        u = p - alfa*t;
        p = t;
        beta = norm(u);
        u = u/beta;
        
        t = v;
        v = q - alfa*t;
        q = t;
        gama = norm(v);
        v = v/gama;
        
        % Form LQ factorization
        gamma =   sqrt( gbar^2 + oldg^2);
        cs    =   gbar / gamma;
        sn    =   oldg / gamma;
        delta =   cs*dbar + sn*alfa;
        gbar  =   sn*dbar - cs*alfa;
        epsln =   sn*beta;
        dbar  = - cs*beta;
        
        % Update x and w
        z = rhs1 / gamma;
        s = z*cs;
        t = z*sn;
        
        x = w*s + q*t + x;
        w = w*sn - q*cs;
        
        % Test for convergene
        rhs1   = rhs2 - delta*z;
        rhs2   =      - epsln*z;
        zbar   = rhs1 / gbar;
        eta    = sn*z - cs*zbar;
        AAnorm = AAnorm + alfa^2 + beta^2 + gama^2;
        Anorm  = sqrt(AAnorm);
        cgnorm = beta*abs(eta);
        qrnorm = qrnorm*sn;
        xnorm  = sqrt( xxnorm + zbar^2 );
        xxnorm = xxnorm + z^2;
        
        test = cgnorm / bnorm;
        t1   = test / (1 + Anorm*xnorm/bnorm);
        rtol = btol + atol*Anorm*xnorm/bnorm;
        t1   = 1 + t1;
        
        if( test < rtol ) istop = 1; end
        if( t1 <= 1 ) istop = 4; end

        if( istop > 0 ) break; end
    end
    
    if( istop == 0 ) istop = 7; end;
    
    x = x + zbar*w;
    rnorm = cgnorm;
end