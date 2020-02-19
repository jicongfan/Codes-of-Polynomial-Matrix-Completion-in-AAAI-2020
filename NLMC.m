function [X]=NLMC(X0,M,p,ker,opt_alg,maxiter)
% The algorithm proposed in
% Non-linear matrix completion. J Fan, TWS Chow. Pattern Recognition 2018, 77, 378-394
% opt_alg can be 'Adam', 'lbfgs', 'cg', ... regure the minFunc toolbox
% for rbf kernel, there is one more paraemter ker.c, can be 0.5,1,2,3, or 5
% ker: kernel typ and parameters, Gaussian RBF (suggested) or polynomial kernel
% for rbf kernel, there is one more paraemter ker.c, can be 0.5,1,2,3, or 5
[d,n]=size(X0);
X=X0;
if strcmp(ker.type,'rbf') && isempty(ker.par)
    XX=sum(X.*X,1);
    D=repmat(XX,n,1) + repmat(XX',1,n) - 2*X'*X;
    if isempty(ker.c)
        ker.c=3;% default
    end
    ker.par=(mean(real(D(:).^0.5))*ker.c)^2;% gamma
end
if strcmp(ker.type,'poly') && isempty(ker.par)
    ker.par=[1 2];% [c d]% default
end
disp(['Kernel type: ' ker.type '; parameter(s):' num2str(ker.par)])   
%%
idx=find(M==0);
%%%% simple initialization: column-row mean
for i=1:d
    Ar(i)=mean(X(i,find(M(i,:)==1)));
end
for i=1:n
    Ac(i)=mean(X(find(M(:,i)==1),i));
end
for i=1:d
    for j=1:n
        A(i,j)=(Ar(i)+Ac(j))/2;
    end
end
x=A(find(M==0));
if length(x)==1
    x=ones(sum(1-M(:)),1)*x;
end
%
V.X0=X0;
V.M=M;
V.p=p;
V.ker=ker;
%% optimization algorithms or Adam
%%%%%% minFunc: [ sd | csd | bb | cg | scg | pcg | {lbfgs} | newton0 | pnewton0 | qnewton | mnewton | newton | tensor ]
if strcmp(opt_alg,'Adam')
    opt.alpha=1e-3;
    opt.maxiter=maxiter;
    [J,x]=opt_Adam(@MC_Kernel_Function,x,V,opt);
else
    options.Method=opt_alg;
    options.MaxIter=maxiter;
    options.TolFun=1e-10;
    options.TolX=1e-10;
    options.Display='iter';
    [x,f,exitflag,output] = minFunc(@MC_Kernel_Function,x,options,V);
end
%%
X(idx)=x;
end
%%
function [f,g]=MC_Kernel_Function(x,V)
%% parameters
X0=V.X0;
M=V.M;
p=V.p;
ker=V.ker;
clear V;
[d,n]=size(X0);
%% kernel
tmp=X0(:);
idx=find(M==0);
tmp(idx)=x;
X=reshape(tmp,d,n);
if strcmp(ker.type,'rbf')
    XX=sum(X.*X,1);
    D=repmat(XX,n,1) + repmat(XX',1,n) - 2*X'*X;
    K=exp(-D/2/ker.par); 
end
if strcmp(ker.type,'poly')
    K=(X'*X+ker.par(1)).^ker.par(2);
end
%% objective function and gradient
% f=trace(K^(p/2));% objective function
% gLgK=p/2*K^(p/2-1);
%%%%%
% [Uk,Sk,~]=lansvd(K,350);Sk=diag(Sk);
% eee=(trace(K)-sum(Sk))/size(Uk,2);
% Sk=Sk+eee*0.99;
% % Uc=null(Uk');
% % Kp2=[Uk Uc]*diag([Sk.^(p/2);ones(size(Uc,2),1)*1e-6])*[Uk Uc]';
% Kp2=Uk*diag(Sk.^(p/2))*Uk';

Kp2=K^(p/2);
f=real(trace(Kp2));% objective function
gLgK=real(p/2*Kp2*(K+eye(n)*1e-6)^(-1));

if strcmp(ker.type,'rbf')
    T=gLgK.*K;
    gLgX=2/ker.par*(X*T-X.*repmat(sum(T),d,1));
%     gLgX=2/ker.par*(X*T-bsxfun(@times,X,sum(T)));
    g=gLgX(idx);% gradient
end

if strcmp(ker.type,'poly')
    T=gLgK;
    gLgX=2*ker.par(2)*X*(T.*(X'*X+ker.par(1)).^(ker.par(2)-1));
    g=gLgX(idx);
end
%
end 



