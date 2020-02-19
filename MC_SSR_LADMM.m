function [X,Z,E]=MC_SSR_LADMM(X0,M,lambda)
% X=XZ+E: Z could be sparse, low-rank or dense; can handle noise and outliers
% This is the code of sparse self representation based matrix completion
% method proposed in the following paper
% Matrix completion by least-square, low-rank, and sparse self-representations
% J Fan, TWS Chow. Pattern Recognition, 2017, 71, 290-305.
[d,n]=size(X0);
%
X=X0;
Z=zeros(n,n);
E=zeros(d,n);
Q=zeros(d,n);
%
maxIter=1000;
u=1e-1;% can be tuned
u_max=1e6;
r0=1.01;% can be tuned
e1=1e-6;
e2=1e-5;
%
normF_X0=norm(X0,'fro');
%
iter=0;
%
%
I=eye(n);
%
while iter<maxIter
    iter=iter+1;
    % Z_new
    eta1=1.02*normest(X)^2;
    temp=Z+X'*(X-X*Z-E+Q/u)/eta1;
%     Z_new=solve_NuclearNorm(temp,u*eta1);
    Z_new=max(0,temp-1/(u*eta1))+min(0,temp+1/(u*eta1));
    Z_new=Z_new-diag(diag(Z_new));
    % X_new
    temp=(X-X*Z_new-E+Q/u)*(I-Z_new)';
    eta2=1.02*norm(I-Z_new)^2;% or normest
%     [~,ttt,~]=lansvd(I-Z_new,10);
%     eta2=1.02*ttt(1,1)^2;
    X_new=X-temp/eta2;
    X_new=X_new.*~M+X0.*M;
    % E_new
    temp=X_new-X_new*Z_new+Q/u;
%     E_new=solve_l1l2(temp,lambda/u);% clolumn-wise outliers
%     E_new=max(0,temp-1/(u/lambda))+min(0,temp+1/(u/lambda));% sparse
%     corruption
    E_new=u/(2*lambda+u)*temp;% dense noise
%     E_new=E_new.*~M;

    %
    Q=Q+u*(X_new-X_new*Z_new-E_new);
    %
    stopC1=norm(X_new-X_new*Z_new-E_new,'fro')/norm(X_new,'fro');
    stopC2=max([norm(Z_new-Z,'fro')/norm(Z,'fro') norm(X_new-X,'fro')/norm(X,'fro') norm(E_new-E,'fro')/norm(E,'fro')]);
    %
    isstopC=stopC1<e1&&stopC2<e2;
    if mod(iter,100)==0||isstopC
        disp(['iteration=' num2str(iter) '/' num2str(maxIter) '  mu=' num2str(u)])
        disp(['stopC1=' num2str(stopC1) '  stopC2=' num2str(stopC2) '  ......'])
    end
    if isstopC
        disp('converged')
        break;
    end
    %
    if stopC2<e2
        r=r0;
    else
        r=r0;
    end 
    u=min(u_max,r*u);
    Z=Z_new;
    X=X_new;
    E=E_new;
end
      
end





