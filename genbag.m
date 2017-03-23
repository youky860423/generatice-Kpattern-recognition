function [X,temp_pos_vec,sk_pos_vec] = genbag( s,m,n )
%GENBAG Generating bag examples of containing the activation signals
%   input:s-template signal;n-number of bags;m-number of instances;
%   output: X is the bag with m instances


%making the bag examples
sigma=1;%%each bags generate each signal
X={};
k=size(s,2);
d=size(s,1);
for i=1:n
    sk_pos=randi(k,1);%%%which template is using for bag i
    temp_pos=randi(m,1); %%%position of the template in bag i
    temp_pos_vec(i,:)=temp_pos;
    sk_pos_vec(i,:)=sk_pos;
    sk=s(:,sk_pos);
    for j=1:m
        if j==temp_pos
            X{i}(:,j)=sk+sigma*randn(d,1);
        else
            X{i}(:,j)=sigma*randn(d,1);
        end
    end
end

% temp_pos_vec

end

