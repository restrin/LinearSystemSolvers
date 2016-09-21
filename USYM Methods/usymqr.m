function [x, istop, itn, rnorm, Anorm, Arnorm] = ...
    usymqr(A,b,c,atol,btol,maxit)

    %------------------------------------------------------------------
    % USYMQR finds a solution x to the system of linear equations Ax = b
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
    %            2       norm(A'*(Ax-b)) is sufficiently small,
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
    itn    = 0;
    istop  = 0;
    
    x = zeros(n,1);
    
    % Set up the first vectors for the tridiagonalization
    
    beta1  = norm(b);
    u     = b/beta1;
    gama1 = norm(c);
    v     = c/gama1;

    p = zeros(m,1);
    q = zeros(n,1);
    beta = 0;
    gama = 0;
    
    AAnorm = 0;
    sbar   = gama;
    rhs1   = beta1;
    tau    = 0;
    gamold = 0;
    cold = 1;
    c = 1; s = 0;
    rnorm = beta1;
    q1 = 0; q2 = 0;
    
    w1 = zeros(n,1);
    w2 = zeros(n,1);
    
    for itn=1:maxit      
        vold = v;
        tauold = tau;
        % Perform next step of tridiagonalization
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

        % Compute useful scalars
        Arnorm = rnorm*norm([gamold*q1 + alfa*q2; gama*q2]);
        Anorm = sqrt(AAnorm);
        AAnorm = AAnorm + alfa^2 + beta^2 + gama^2;
        
        if (Arnorm/(Anorm*rnorm) < atol) istop = 2; end
        
        % Form QR factorization   
        sigma  =   c*sbar + s*alfa;
        rbar   = - s*sbar + c*alfa;
        tau    =            s*gama;
        sbar   =            c*gama;

        if( itn == 1 )
            rbar = alfa;
            sbar = gama;
        end
        
        rho = sqrt(rbar^2 + beta^2);
        c = rbar/rho;
        s = beta/rho;
        
        t = rhs1;
        rhs1 =  c*t;
        rhs2 = -s*t;
        
        % Update solution
        w3 = ( vold - sigma*w2 - tauold*w1 )/rho;
        
        x = x + rhs1*w3;
        w1 = w2; w2 = w3; 

        % Check convergence criteria
        rnorm = abs(rhs2);
        q1 = -cold*s;
        q2 = c;
        
        if( istop > 0 ) break; end
        % If converged due to residual small enough,
        % do one extra half-step to get correct estimate of ||A'*(A*x-b)||
        if( rnorm < atol*Anorm + btol ) istop = 1; end
        
        rhs1 = rhs2;
        gamold = gama;
        cold = c;
    end
    
    if( istop == 0 ) istop = 7; end;
end