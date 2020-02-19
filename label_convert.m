function Y=label_convert(L)
u=unique(L);
n=length(L);
c=length(u);
Y=zeros(c,n);
for i=1:n
    j=find(u==L(i));
    Y(j,i)=1;
end
end
