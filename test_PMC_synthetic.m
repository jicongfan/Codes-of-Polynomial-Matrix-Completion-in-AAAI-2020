clc
clear all
warning off
% nk=[6:1:10];
missrate=[0.5];% fraction of missing data
for pp=1:1
% generate data
ns=5;
m=20;
X=[];
n=50;
r=2;
rho=0.5;
for k=1:ns
    x=unifrnd(-1,1,[r,n]);
    T=randn(m,r)*x+rho*(1*randn(m,r)*x.^2+randn(m,r)*x.^3+1*randn(m,r)*x.^4);
    X=[X T];
end
% produce missing data
[nr,nc]=size(X);
M=ones(nr,nc);
for i=1:nc
    temp=randperm(nr,ceil(nr*missrate));% 1
    M(temp,i)=0;
end
% rank(X)
X0=X;
X=X.*M;
%% low-rank matrix completion via nuclear norm minimization
[Xr{1}]=LRMC_NNM(X,M);
%% Sparse self-representtion based matrix completion
% [Xr{2},Z,E_SSR]=MC_SSR_LADMM(X,M,100/missrate);
%% VMC Ongie's method; need to download the code 
% sampmask=logical(M);
% samples=X0(sampmask);   
% options.d=3;options.c=10;options.niter=1000;options.p=0.5;
% tic;[Xr{3},cost,update,error] = vmc(X,sampmask,samples,options,X0);t(3)=toc;
% options.d=2;options.c=1;options.niter=1000;options.p=0.5;
% tic;[Xr{4},cost,update,error] = vmc(X,sampmask,samples,options,X0);t(3)=toc;
%% NLMC: method in the paper of Fan and Chow 2018
ker.type='rbf';ker.par=[];ker.c=3;[Xr{2}]=NLMC(X,M,0.5,ker,'Adam',500);
%% PMC-S
s=nr;
ker.type='rbf';ker.par=[];ker.c=3;[Xr{3}]=PMC_S(X,M,0.5,s,ker,500);t(5)=toc;%
%% PMC-W
w=[];% [] for auto-designed or fixed value such as ([1:nc]'/nc).^1
tic;ker.type='rbf';ker.par=[];ker.c=3;[Xr{4}]=PMC_W(X,M,0.5,w,ker,500);t(5)=toc;% 
%%
for i=1:length(Xr)
    if ~isempty(Xr{i})
    	MSE(pp,i)=norm((X0-Xr{i}).*(1-M),'fro')/norm(X0.*(1-M),'fro');
    end
end
end
disp(MSE)


