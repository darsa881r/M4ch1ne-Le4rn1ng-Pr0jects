close all
clearvars

L =100; % No of experiment datasets
K =120; % No of relaxation factors used
lambda = linspace(exp(7),exp(8),K);

%% Generating the training set

%Initiating a random number generator
rng(0,'v4');

D = 1;
N_train = 25;
x_train = zeros(N_train,L); 

std = 0.30;
mean = 0;
e_train = zeros(N_train,L); 

t_train = zeros(N_train,L); 

for i =1:L

    % rand by default creates a uniform dist. from 0 to 1
    x_train(:,i) = rand(N_train,D);
    % randn by default creates a normal dist. from 0 to 1 with mean 0 and var 1
    e_train(:,i) = std.*randn(N_train,D) + mean;
    
    t_train(:,i) = sin(2*pi*x_train(:,i)) + e_train(:,i);
    
%     figure
%     scatter(x_train(:,i), t_train(:,i), 'bo')
    
end


%% Generating the test set

N_test = 1000;

% rand by default creates a uniform dist. from 0 to 1
x_test = rand(N_test,D);

% randn by default creates a normal dist. from 0 to 1 with mean 0 and var 1
std = 0.30;
mean = 0;

e_test = zeros(N_test,L); 

t_test = zeros(N_test,L); 

for i =1:L

    % rand by default creates a uniform dist. from 0 to 1
    x_test(:,i) = rand(N_test,D);
    % randn by default creates a normal dist. from 0 to 1 with mean 0 and var 1
    e_test(:,i) = std.*randn(N_test,D) + mean;
    
    t_test(:,i) = sin(2*pi*x_test(:,i)) + e_test(:,i);
    
%     figure
%     scatter(x_train(:,i), t_train(:,i), 'bo')
    
end 
% figure
% scatter(x_test, t_test, 'rx')

%% Generating Gaussian Basis function Design Matrix with x_train data and predictors

%Ideal sinusoidal
S=100;
x1=linspace(0,1,S);
y1 = sin(2*pi*x1);

phi(:,:,L) =zeros(N_train,N_train);
phi_test(:,:,L) =zeros(N_test,N_train);
W(:,:,L) = zeros(N_train,K);
y2(:,:,L) = zeros(N_train,K);
y2_test(:,:,L) = zeros(N_test,K);


for j=1:K
    for m =1:L
        % Generating Design matrix, Basis function
        phi(:,:,m) =zeros(N_train,N_train);
        phi(:,1,m) =ones(N_train,1);
        phi_test(:,:,m) =zeros(N_test,N_train);
        phi_test(:,1,m) =ones(N_test,1);

        sigma = 0.10;

        for i = 1:(N_train-1)
            z = ((x_train(:,m) - x_train(i,m))/(sqrt(2)*sigma));
            phi(:,i+1,m) = exp(-z.^2) ;
        end

        for i = 1:(N_train-1)
            z_test = (x_test(:,m) - x_train(i,m))/(sqrt(2)*sigma);
            phi_test(:,i+1,m) = exp(-z_test.^2) ;
        end

        A = phi(:,:,m)'*phi(:,:,m) + lambda(j) * eye(N_train);
        B = phi(:,:,m)'* t_train(:,1);
        W(:,j,m) = A\B;

        y2(:,j,m) =  W(:,j,m)'*phi(:,:,m)';
        y2_test(:,j,m) =  W(:,j,m)'*phi_test(:,:,m)';


    end

end

%% Statistical approach Bias - Variance

sum_f(:,:,1) = zeros(N_train,K);
for i=1:L
    sum_f(:,:,1) = y2(:,:,i)+ sum_f(:,:,1);
end
avg_f = sum_f/L;

% dev22  =  (avg_f-repmat((sum(t_train,2)/L),1,K)).^2;
% bias22 = sum(dev22,1)./N_train;

% using H = sin(2pix)

H = sin(2*pi*x_train);
avg_H = sum(H,2)/L;
dev22  =  (avg_f-repmat(avg_H,1,K)).^2;
bias22 = sum(dev22,1)./N_train;

%% Generating Variance

dev3(:,:,L)=zeros(N_train,K);
sum_v1(:,:,1)=zeros(N_train,K);

for j=1:L
    dev3(:,:,i) = (y2(:,:,i)-avg_f).^2;
    sum_v1(:,:,1) = dev3(:,:,i)+ sum_v1(:,:,1);
end

var=sum(sum_v1(:,:,1),1)./N_train;

total=bias22+var;

%% Finding Test Error

avg_t_test = sum(t_test,2)./N_test;

sum_yt(:,:,1) = zeros(N_test,K);

for i=1:L
    sum_yt(:,:,1) = y2_test(:,:,i)+ sum_yt(:,:,1);
end

avg_yt = sum_yt/L;

test_error = (avg_yt - repmat(avg_t_test,1,K)).^2;

avg_TE = (sum(test_error,1)./N_test)+total;


%% Results

figure
plot(log(lambda),bias22)
hold on
plot(log(lambda),var)
plot(log(lambda),total)
plot(log(lambda),avg_TE)
hold off
title('Bias Variance Trade-off');
legend('(bias)2', 'variance', 'total error','test error');























