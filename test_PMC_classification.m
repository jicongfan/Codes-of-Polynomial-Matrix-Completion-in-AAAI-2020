clc
clear all
for pp=1:1
load('mice.mat');    F=X;    L=Label; 
% load('dermatology.mat')% dermatology  satimage   shuttle
% F=Xorg(:,1:end-1)';
% L=Xorg(:,end)';
ids=randperm(length(L),500); F=F(:,ids); L=L(:,ids);% use a subset for faster demo
label_NAN=find(isnan(L));
F(:,label_NAN)=[];
L(:,label_NAN)=[];
L=label_convert(L);
M_org=1-double(isnan(F));
obv_ratio=sum(M_org(:))/prod(size(M_org))
F(isnan(F))=0;
% normalize if necessary
for i=1:size(F,1)
    id1=find(M_org(i,:)==1);
    x=F(i,id1);
    F(i,:)=(F(i,:)-mean(x))./std(x);
end
%%
label_train=0.5;% proportion of training data, can be tuned
feature_missrate=0.5;% proportion of missing values in the feature matrix, can be tuned
%%
X_F=F;
X_L=L;
[nr,nc]=size(X_F);
M_t=ones(nr,nc);
for i=1:nc
    id0=find(M_org(:,i)==1);
    lid0=length(id0);
    temp=randperm(lid0,ceil(lid0*feature_missrate));
    temp=id0(temp);
    M_t(temp,i)=0;
end
M_F=M_t.*M_org;
id_train=randperm(nc,round(label_train*nc));
id_test=setdiff(1:nc,id_train);
M_L=ones(size(X_L));
M_L(:,id_test)=0;
%
X=[X_F;X_L];
M=[M_F;M_L];
D=X.*M;
k=0;
%%
k=k+1;
tic
[Dc{k},E]=LRMC_NNM(D,M);
Yc{k}=Dc{k}(size(X_F,1)+1:end,:);
T(k)=toc;
method_name{k}='LMC-IALM';
%%
k=k+1;
ker.type='rbf';ker.par=[];ker.c=3;[Dc{k}]=NLMC(D,M,0.5,ker,'Adam',500);
Yc{k}=Dc{k}(size(X_F,1)+1:end,:);
T(k)=toc;
%%
k=k+1;
s=5;
ker.type='rbf';ker.par=[];ker.c=3;[Dc{k}]=PMC_S(D,M,0.5,s,ker,500);
Yc{k}=Dc{k}(size(X_F,1)+1:end,:);
%%
k=k+1;
% w=([1:nc]'/nc).^1;
w=[];
ker.type='rbf';ker.par=[];ker.c=3;[Dc{k}]=PMC_W(D,M,0.5,w,ker,500);
Yc{k}=Dc{k}(size(X_F,1)+1:end,:);
%%
for i=1:length(Dc)
    [tmp,L_0]=max(L(:,id_test));
    [tmp,L_c]=max(Yc{i}(:,id_test));
    c_error(pp,i)=sum((L_c-L_0)~=0)/length(L_0);
end
%% SVM need the SVM-KM matlab toolbox
% % X_train=X_F(:,id_train).*(M(1:nr,id_train));
% % X_test=X_F(:,id_test).*(M(1:nr,id_test));
% % [tmp,L_train]=max(L(:,id_train));
% % [L_train, L_test]=label_unify(L_train,L_0);
% % C=1000;
% % epsilon=1e-4;
% % verbose=0;
% % kernel='gaussian';% gaussian
% % kerneloption=100;% poly: 1 better % gaussian 1e4
% % kerneloptionm.matrix=svmkernel(X_train',kernel,kerneloption);
% % [xsup,w,b,nbsv,classifier,pos]=svmmulticlassoneagainstone([],L_train',length(unique(L_train)),C,epsilon,'numerical',kerneloptionm,verbose);
% % kerneloptionm.matrix=svmkernel(X_test',kernel,kerneloption,X_train(:,pos)');
% % [L_predic_svm,maxi] = svmmultivaloneagainstone([],[],w,b,nbsv,'numerical',kerneloptionm);
% % c_error(pp,i+1)=sum((L_predic_svm'-L_test)~=0)/length(L_test);
%% LRMC+SVM need the SVM-KM matlab toolbox
% % X_train=Dc{1}(1:nr,id_train);
% % X_test=Dc{1}(1:nr,id_test);
% % C=1000;
% % kernel='gaussian';% gaussian
% % kerneloption=100;% poly: 1 better % gaussian 1e4
% % kerneloptionm.matrix=svmkernel(X_train',kernel,kerneloption);
% % [xsup,w,b,nbsv,classifier,pos]=svmmulticlassoneagainstone([],L_train',length(unique(L_train)),C,epsilon,'numerical',kerneloptionm,verbose);
% % kerneloptionm.matrix=svmkernel(X_test',kernel,kerneloption,X_train(:,pos)');
% % [L_predic_svm,maxi] = svmmultivaloneagainstone([],[],w,b,nbsv,'numerical',kerneloptionm);
% % c_error(pp,i+2)=sum((L_predic_svm'-L_test)~=0)/length(L_test);

end
disp(c_error)


