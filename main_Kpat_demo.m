clear all
close all
%activation template generating or pattern matching%%%%%
k=3;%number of pattern;
d=100;%%%dimension of the signal;
s(:,1)=5*[ones(d/2,1);-ones(d/2,1)];
s(:,2)=(1:d)'/10;
s(:,3)=(-1)*(1:d)'/10;
m=100;%total number of instances,
n=50;%total number of bags,
[X,temp_pos_vec,sk_pos_vec] = genbag( s,m,n );
%%%%%%plotting data%%%%%%%%
figure(2)
plot(s);
title('Ture pattern');
figure(1)
for i=1:n 
    plot(X{i})
    title(['bag ', num2str(i)]);
    pause(0.1)
end
figure(4)
plot(sk_pos_vec,'b*-')
title('label distribution for bags')
%%%%%%%%%%learning model%%%%%%%%%
mitr=10;
for iter=1:mitr
    alpha_init=rand(1,k);
    alpha_init=alpha_init/sum(alpha_init);
    sinit=2*randn(d,k);
    [ label(:,iter), model{iter}, llh] = emmgms(X,alpha_init,sinit);
    maxllh(iter,:)=llh(end);
end
[mxlh,maxidx]=max(maxllh);
maxmodel=model{maxidx};
maxlabel=label(:,maxidx);
stmp=zeros(size(s));
ltmp=zeros(size(maxlabel));
%%%%%%%%%shifting the order of the learned template to match%%%%%
for k1=1:k
    dist=zeros(k,1);
    for k2=1:k
        dist(k2)=norm(maxmodel.s(:,k2)-s(:,k1));
    end
    [~,midx]=min(dist);
    stmp(:,k1)=maxmodel.s(:,midx);
    ltmp(maxlabel==midx)=k1;
end
maxmodel.s=stmp;
maxlabel=ltmp;
figure(3)
plot(maxmodel.s);
title('learned K pattern');
figure(4)
hold on
plot(maxlabel,'ro-')
hold off
legend('True label','learned label');