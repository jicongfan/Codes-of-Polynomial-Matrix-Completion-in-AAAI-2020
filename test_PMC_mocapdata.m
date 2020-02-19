clc
clear all
warning off
missrate=0.5;
for pp=1:1
load mocap_56_01.mat
X=X(:,1:4:end);% down-sampling
%%    
N=size(X,2);
% miss data
[nr,nc]=size(X);
M=ones(nr,nc);
for i=1:N
    temp=randperm(nr,round(nr*missrate));% 1
    M(temp,i)=0;
end
X0=X;
X=X.*M;
%%
%% LRMC & SRMC
[Xr{1}]=LRMC_NNM(X,M);%
% [Xr{2}]=MC_SSR_LADMM(X,M,1000);
%% VMC Ongie's method; need to download the code  
% sampmask=logical(M); samples=X0(sampmask);
% options.d=2;options.c=1;options.niter=1000;options.p=0.5;
% [Xr{3},cost,update,error] = vmc(X,sampmask,samples,options,X0);
% options.d=3;options.c=1;options.niter=1000;options.p=0.5;
% [Xr{4},cost,update,error] = vmc(X,sampmask,samples,options,X0);
%% NLMC
ker.type='rbf';ker.par=[];ker.c=5;[Xr{2}]=NLMC(X,M,0.5,ker,'Adam',500);
%% PMC
ker.type='rbf';ker.par=[];ker.c=5;[Xr{3}]=PMC_S(X,M,0.5,round(nr/2),ker,500);
ker.type='rbf';ker.par=[];ker.c=5;[Xr{4}]=PMC_W(X,M,0.5,[],ker,500);
%%
for i=1:length(Xr)
    if ~isempty(Xr{i})
        MAE(pp,i)=sum(sum(abs(X0-Xr{i}).*(1-M)))/sum(sum(abs(X0).*(1-M)));
    end
end
end
disp(MAE)

