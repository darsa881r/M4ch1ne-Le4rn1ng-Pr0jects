clearvars
close all
clc
K = 5;
N = 200;
for i = 1:K
I{i} = crossvalind('Kfold',(N/K),N);
end

