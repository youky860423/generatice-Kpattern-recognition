function   [s_initial]=initials(Y,weight,phi)
%%%%%%Alignment of bags of signals with k distinct pattern%%%%%%

%%%%parameter settings:%%%%%%%
N=length(Y);
Wnk=weight./(ones(N,1)*sum(weight,1));%normalized weight
k=size(weight,2);

%%%%%first step: initialization%%%%%%
idx_vec1=zeros(N,1);
val_vec1=zeros(N,1);
%MMM=zeros(N,N);
for l=1:k

    for i=1:N
%         disp('initial s at')
%         i
        %Dij_tot=zeros(no_delays,no_delays);
        no_delaysi=size(Y{i},2);
        v=zeros(no_delaysi,1);
        tmpidx=1:N;
        tmpidx(i)=[];
        wi=Wnk(i,l);
        for j=tmpidx;
            no_delaysj=size(Y{j},2);
            Dij=sum(Y{i}.^2)'*ones(1,no_delaysj) - 2*Y{i}'*Y{j} + ones(no_delaysi,1)*sum(Y{j}.^2) + ...
                ones(no_delaysi,1)*phi{j}';
            wj=Wnk(j,l);
            v=v+min(wj*Dij')';
            [ftaui{i,j},idxtaui{i,j}]=min(wj*Dij');
        end
        [val_vec1(i),idx_vec1(i)]=min(v+wi*phi{i}); %%% min_k min x_i \sum_{j in{1:N}\{k}} min x_j x_i' Dij x_j
       % MMM(:,i)=MM(:,i); %%% the ith column of MMM represents the optimal solution to
        %%% minimizing \sum_{j \neq i} D(Yi xi, Yj xj)^2
        %MMM(i,i)=idx_vec(i);

    end
    [~,idx]=min(val_vec1);
    %pause
    i_opt(l)=idx;
    yi_opt(l)=idx_vec1(idx);
    %%%%%calculating optimal instances in each bag j
    if N==1
        y_opt_vec(:,l)=yi_opt;
    else
        for j=1:N
            if ~isempty(idxtaui{i_opt(l),j})
                y_opt_vec(j,l)=idxtaui{i_opt(l),j}(yi_opt(l));
            else
                y_opt_vec(j,l)=0;
            end
        end
        y_opt_vec(i_opt,l)=yi_opt(l);
    end
end


%%%%%%step two: compute the error bound for objective%%%%%%%
% tmp1=0;tmp2=0;
% for i=1:N
%     for j=1:N
%         if ~isempty(idxtaui{i,j})
%             taui_opt(j)=idxtaui{i,j}(idx_vec1(i));
%         else
%             taui_opt(j)=0;
%         end
%     end
%     taui_opt(i)=idx_vec1(i);
%     tmp3=0;
%     for j=1:N
%         tmp1=tmp1+norm(Y{i}(:,tau_opt_vec(i))-Y{j}(:,tau_opt_vec(j)))^2;
%         tmp3=tmp3+norm(Y{i}(:,taui_opt(i))-Y{j}(:,taui_opt(j)))^2+phi{j}(taui_opt(j));
% %         for k=1:no_delays
% %             tmp4(i)=norm(Y{i}(:,tau_opt_vec(i))-Y{j}(:,k))^2;
% %         end
%     end
%     fi(i)=tmp3;
%     tmp2=tmp2+phi{i}(tau_opt_vec(i));
% end
% %%% checking for the bounds%%%%
% f_aml(1)=tmp1/(2*N)+tmp2
% f_lower(1)=1/2*mean(val_vec1)
% f_upper(1)=val
%%%%%step three:for generating activation signature%%%%%
d=size(Y{1},1);
for l=1:k
y_star=zeros(d,1);
    for i=1:N;
        y_star=y_star+Wnk(i,l)*Y{i}(:,y_opt_vec(i,l));
%         figure(l)
%         plot(Y{i}(:,y_opt_vec(i,l)))
%         hold on
    end
%     hold off
    s_initial(:,l)=y_star;
% %    pause()
end
% % figure(1);
% % if k==1
% %     plot(1:d,s_initial(:,1),'b')
% %     title('initialization for sk during the M step')
% % end
% % if k==2
% %     plot(1:d,s_initial(:,1),'b',1:d,s_initial(:,2),'r')
% %     title('initialization for sk during the M step')
% % end
% % if k==3
% %     plot(1:d,s_initial(:,1),'b',1:d,s_initial(:,2),'r',1:d,s_initial(:,3),'g')
% %     title('initialization for sk during the M step')
% % end

end

