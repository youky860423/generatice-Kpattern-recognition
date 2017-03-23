function [ phi ] = genphi( Y )
%%%% data generation (construct phi's from Yi's)


for i=1:length(Y)
    temp=max(sum(Y{i}.^2))-sum(Y{i}.^2);
    phi{i}=temp';
end

