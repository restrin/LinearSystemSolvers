function [] = blah()
n = 100;
m = 60;
A = rand(n);
G1 = rand(m,n);
G2 = rand(m,n);
g = rand(m,1);

A = A+1e1*eye(n);
G2 = G1;
G2 = (G2*(A\G1') + 1e-1*rand(m))\G2;

tol = 1e-10;

AA = @AAA;
GG1 = @GGG1;
GG2 = @GGG2;

K = [A G1';G2 zeros(m)];
soln = K\[zeros(n,1);g];
xsoln = soln(1:n);
ysoln = soln(n+1:end);
maxiter = 2*m;


%%


%%

[x,y,iter,resvec] = spqmrsc(AA, GG1, GG2, g, tol, maxiter,[]);

iter
%resvec
%K*[x;y] - [zeros(n,1);g]
[ norm(K*[x;y] - [zeros(n,1);g]) norm(x - xsoln) norm(y-ysoln)]

[x,y,iter,resvec] = spmrsc(AA, GG1, GG2, g, tol, maxiter,[]);

iter
%resvec
%K*[x;y] - [zeros(n,1);g]
[ norm(K*[x;y] - [zeros(n,1);g]) norm(x - xsoln) norm(y-ysoln)]


f = rand(n,1);

K = [A G1';G2 zeros(m)];
soln = K\[f;zeros(m,1)];
xsoln = soln(1:n);
ysoln = soln(n+1:end);
maxiter = 2*m;

ZZ1 = @Z1;
ZZ2 = @Z2;

[ x, iter, resvec ] = spmrns( A, ZZ1, ZZ2, f, tol, maxiter, [] );

iter
%resvec
norm(x-xsoln)

[ x, iter, resvec ] = spqmrns( A, ZZ1, ZZ2, f, tol, maxiter, [] );

iter
%resvec
norm(x-xsoln)

% [ x, y, iter, resvec ] = ULSQR_sym( A, G1, G2, g, g, tol, 20, eye(m), 1 );
% 
% iter
% resvec
% K*[x;y] - [zeros(n,1);g]
% norm(x - xsoln)
% norm(y-ysoln)

% [ x, y, iter, resvec, errvec, resvece ] = ...
%     ULSQR( A, G1, G2, g, g, tol, 20, eye(m), 1, xsoln, ysoln )
% 
% iter
% resvec
% K*[x;y] - [zeros(n,1);g]
% norm(x - xsoln)
% norm(y-ysoln)

function r = AAA(x,t)
    if t==1
        r = A\x;
    else
        r = A'\x;
    end
end

    function r = GGG1(x,t)
        if t==1
            r = G1*x;
        else
            r = G1'*x;
        end
    end

    function r = GGG2(x,t)
        if t==1
            r = G2*x;
        else
            r = G2'*x;
        end
    end

    function r = Z1(x,t)
        r = [eye(n) G1';G1 zeros(m)]\[x;zeros(m,1)];
        r = r(1:n);
    end

    function r = Z2(x,t)
        r = [eye(n) G2';G2 zeros(m)]\[x;zeros(m,1)];
        r = r(1:n);
    end

end