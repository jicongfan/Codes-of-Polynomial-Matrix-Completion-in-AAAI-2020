function [L10,L20]=Label_unify(L1,L2)
Y=unique(L1);
L10=[];
for k=1:length(Y)
    idx=find(L1==Y(k));
    L10(idx)=k;
    idx=find(L2==Y(k));
    L20(idx)=k;
end
end
    