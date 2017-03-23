function [label,model, llh] = emmgms(X,alpha_init,sinitial)
% EM algorithm for fitting the mixture of Gaussian mixture model.
%   X: 1 x N cell format data with bag number N; each bag contains instances 
%      d x n(dimension of data and number of instances in the bag)
%   init: k (1 x 1) or label (1 x n, 1<=label(i)<=k) or center (d x k)
%
%% initialization
phi = genphi( X );
tic
fprintf('EM for K mixture of patterns is running ... \n');
% Pik = initialization(X,init);
N=length(X);
k=size(sinitial,2);
d=size(sinitial,1);
sigma=1;
% alpha_init=rand(1,3);
% alpha_init=alpha_init/sum(alpha_init);
% sinitial=2*randn(d,k);
for i = 1:k
    logG(:,i) = loggaussmixpdf(X,sinitial(:,i),sigma);   
end
logG = bsxfun(@plus,logG,log(alpha_init));
T = logsumexp(logG,2);
llh(1) = sum(T)/N; % loglikelihood
logR = bsxfun(@minus,logG,T);
Pik = exp(logR);
[~,label(1,:)] = max(Pik,[],2);
% Pik = Pik(:,unique(label));
maxlogR=max(logR);
Qik=exp(logR-ones(size(logR,1),1)*maxlogR);
Qik=Qik./(ones(size(logR,1),1)*sum(Qik));

tol = 1e-20;
maxiter = 100; %500;
% llh = -inf(1,maxiter);
%%%%%%calculate the first iteration of complete likelihood function%%%%
converged = 0;
t = 1;
toc
while ~converged && t < maxiter
% while  t < maxiter
tic
    t = t+1;
    %slast=initials(X,Pik);
    slast=initials(X,Qik,phi);
toc
    %model = maximization(X,Pik,slast,sigma);
    tic
    model = maximization(X,logR,slast,sigma);
%     model.weight%%do not display weight prob.
    toc
    tic
    [logR, llh(t)] = expectation(X,model);
    toc
    maxlogR=max(logR);
    Pik=exp(logR);
    Qik=exp(logR-ones(size(logR,1),1)*maxlogR);
    Qik=Qik./(ones(size(logR,1),1)*sum(Qik));
    [~,label(:)] = max(Pik,[],2);
%     u = unique(label);   % non-empty components
%     if size(alpha,2) ~= size(u,2)
%         alpha = alpha(:,u);   % remove empty components
%     else
%         converged = llh(t)-llh(t-1) < tol*abs(llh(t));
%     end
    converged = llh(t)-llh(t-1) < tol*abs(llh(t));

    pause(0.1)
 
end
% llh = llh(2:t);
figure(5);plot(llh);title('complete likelihood of emmgms');

if converged
    fprintf('Converged in %d steps.\n',t-1);
else
    fprintf('Not converged in %d steps.\n',maxiter);
end


function [ logR, llh] = expectation(X, model)
s = model.s;
w = model.weight;
sigma=model.sigma;

N = length(X);
k = size(s,2);
logG = zeros(N,k);

for i = 1:k
    logG(:,i) = loggaussmixpdf(X,s(:,i),sigma);   
end
logG = bsxfun(@plus,logG,log(w));
T = logsumexp(logG,2);
llh = sum(T)/N; % loglikelihood
logR = bsxfun(@minus,logG,T);
%R = exp(logR);


function model = maximization(X,logR,slast,sigma)

    maxlogR=max(logR);
    Pik=exp(logR);
    Qik=exp(logR-ones(size(logR,1),1)*maxlogR);
    Qik=Qik./(ones(size(logR,1),1)*sum(Qik));


N=length(X);
d=size(X{1},1);

nk = sum(Pik,1);
w = nk/N;
eps=1e-8;
w=(w+eps)/(1+length(w)*eps);
%%%%%%estimating s using optimization transfer%%%%
k=size(logR,2);
[w3,shat, ~] = MLEforS( X,slast,logR,sigma);
% shat
% w3
% % figure(4);
% % if k==1
% %     plot(1:d,shat(:,1),'b')
% %     title('estimation for sk during the M step')
% % end
% % if k==2
% %     plot(1:d,shat(:,1),'b',1:d,shat(:,2),'r')
% %     title('estimation for sk during the M step')
% % end
% % if k==3
% %     plot(1:d,shat(:,1),'b',1:d,shat(:,2),'r',1:d,shat(:,3),'g')
% %     title('estimation for sk during the M step')
% % end


model.s = shat;
model.sigma = sigma;
model.weight = w;

function y = loggaussmixpdf(X, s,sigma)
N=length(X);
d = size(X{1},1);

for i=1:N
    ni=size(X{i},2);
    f1(i,:)=ni*(d/2)*log(2*pi*sigma^2)+sum(sum(X{i}.^2/(2*sigma^2),1));
    f2(i,:)=-log(ni)+logsumexp((s'*X{i}/(sigma^2)),2);
end
y = -(s'*s)/(2*sigma^2)-f1+f2;


