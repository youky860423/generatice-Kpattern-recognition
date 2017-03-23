function [ w3,shat, obj] = MLEforS( X,sini,logR,sigma)
    maxlogR=max(logR);
    Pik=exp(logR);
    Qik=exp(logR-ones(size(logR,1),1)*maxlogR);
    Qik=Qik./(ones(size(logR,1),1)*sum(Qik));


%MLE Maximum likihood estimation of template S
%input: X-data;weight-membership probability;sini-initial s;s-true value.   


%%%%%initialization%%%%%%%%%
% sini=randn(2*d,1);
k=size(sini,2);
shat=sini;
n=length(X);
d=size(X{1},1);
w3={};
nitr=100;
%%%%%using optimization transfer for optimization%%%%
for i=1:n
    m=size(X{i},2);
    score=X{i}'*shat-ones(m,1)*max(X{i}'*shat,[],1);
    maxscore(i,:)=max(X{i}'*shat,[],1);
    w{i}=exp(score/sigma^2);
    w1(i,:)=sum(w{i},1);%%sum weights over m
    w2=w{i}./(ones(m,1)*sum(w{i},1));% Wijk
    wgts(i,:,:)=(ones(d,1)*Qik(i,:)).*shat;
    wgtdata(i,:,:)=(X{i}*w2).*(ones(d,1)*Qik(i,:));
end 
 w3{1}=wgtdata;
grad{1}=(1/sigma^2)*(reshape(sum(wgts,1)-sum(wgtdata,1),d,k));
for j=1:k
    f(1,j)=-sum(Pik(:,j))*1/(2*sigma^2)*(shat(:,j)'*shat(:,j))+...
           sum(Pik(:,j).*(log(w1(:,j))+maxscore(:,j)));
    shat_vec(1,j)=norm(shat(:,j));
end
shat=reshape(sum(wgtdata,1),d,k);
%%%%%%gradient descent%%%%%%
%     while norm(totgrad,'fro') > eps
  for iter=2:nitr
        %%%calculate gradient%%%%
        for i=1:n
            m=size(X{i},2);
            score=X{i}'*shat-ones(m,1)*max(X{i}'*shat,[],1);
            maxscore(i,:)=max(X{i}'*shat,[],1);
            w{i}=exp(score/sigma^2);
            w1(i,:)=sum(w{i},1);%%sum weights over m
            w2=w{i}./(ones(m,1)*sum(w{i},1));% Wijk
            wgts(i,:,:)=(ones(d,1)*Qik(i,:)).*shat;
            wgtdata(i,:,:)=(X{i}*w2).*(ones(d,1)*Qik(i,:));
        end 
        w3{iter}=wgtdata;
        grad{iter}=(1/sigma^2)*(reshape(sum(wgts,1)-sum(wgtdata,1),d,k));
        for j=1:k
            f(iter,j)=-sum(Pik(:,j))*1/(2*sigma^2)*(shat(:,j)'*shat(:,j))+...
                   sum(Pik(:,j).*(log(w1(:,j))+maxscore(:,j)));
            shat_vec(iter,j)=norm(shat(:,j));
        end
        shat=reshape(sum(wgtdata,1),d,k);
   end
% % for i=1:k
% %     figure(2)
% %     plot(f(:,i))
% %     title('objective function')
% %     figure(3)
% %     plot(shat(:,i))
% %     title('estimated template')
% %     pause()
% % end
% %%%%mean square error on estimation%%%
% Error=(shat-s)'*(shat-s);
obj=sum(f,2);
% % figure(2);plot(obj);title('sum over k on objective for maximizing sk')
% % figure(3);
% % if k==1
% %     plot(1:nitr,f(:,1),'b*');title('objective for maximizing sk')
% % end
% % if k==2
% %     plot(1:nitr,f(:,1),'b*',1:nitr,f(:,2),'ro');title('objective for maximizing sk')
% % end
% % if k==3
% %     plot(1:nitr,f(:,1),'b*',1:nitr,f(:,2),'ro',1:nitr,f(:,3),'gd');title('objective for maximizing sk')
% % end
% % 
% % end
% % 
