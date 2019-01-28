close all
clearvars
%Initiating a random number generator
rng(0,'v4');

%% Generating the training set

N_train = 100;

% rand by default creates a uniform dist. from 0 to 1
x_train = rand(N_train,1);

% randn by default creates a normal dist. from 0 to 1 with mean 0 and var 1
std = 0.50;
mean = 0;
e_train = std.*randn(N_train,1) + mean;


t_train = sin(2*pi*x_train) + e_train;

%  figure
%  scatter(x_train, t_train, 'bo')

%% Generating the test set

N_test = 100;

% rand by default creates a uniform dist. from 0 to 1
x_test = rand(N_test,1);

% randn by default creates a normal dist. from 0 to 1 with mean 0 and var 1
std = 0.50;
mean = 0;
e_test = std.*randn(N_test,1) + mean;


t_test = sin(2*pi*x_test) + e_test;
 
%  figure
%  scatter(x_test, t_test, 'rx')

%% Generating m-order plynomial Design Matrix with x_train data

max_order = 9; %order of the polynomial

ERMS_train = zeros((max_order+1),1);
ERMS_test = zeros((max_order+1),1);

lambda = 0.000; %Ridge regularization Parameter

for m =0:max_order
    
    % Generating Design matrix, Basis function
    phi = zeros(N_train,(m+1));
    for i = 1:m+1
        phi(:,i) = x_train.^(i-1);
    end

    % Using Closed Form equation
    A = phi'*phi + lambda * eye(m+1);
    B = phi'* t_train;
    W1 = A\B;
    
    %initializing variables
    S=400;
    x1=linspace(0,1,S);
    xx1 = zeros(S,m);
    %xx2 = zeros(N_train,m);
    xx_test = zeros(N_test,m);
    
   
    for i = 1:m+1
        xx1(:,i) = x1.^(i-1); %for the continuous curve
        xx_test(:,i) = x_test.^(i-1); %for the test set
    end
    y1 = W1'*xx1';
    y2 = W1'*phi';
    y_test = W1'*xx_test';
    
    %calculating training error
    J_train = (y2 - t_train')*(y2 - t_train')';
    ERMS_train(m+1) = sqrt(J_train/(N_train-1)); 
    
    %calculating testing error
    J_test = (y_test - t_test')*(y_test - t_test')';
    ERMS_test(m+1) = sqrt(J_test/(N_test-1));
    
    figure
    scatter(x_train, t_train, 'bo')
    hold on
    plot(x1,y1','b');
    hold off
    title(['Curve Fitting for order ',num2str(m)])
    xlabel('x_train') % x-axis label
    ylabel('t_train') % y-axis label
    legend('Datapoints','Predicted Values')
    
    figure
    scatter(x_test, t_test, 'rx')
    hold on
    plot(x1,y1','g');
    hold off
    title('Synthetic Dataset')
    xlabel('x_test') % x-axis label
    ylabel('t_test') % y-axis label
    legend('Datapoints','Predicted Values')
 
end

figure
plot((0:(max_order))', ERMS_train, '--bo')
hold on
plot((0:(max_order))', ERMS_test, '--ro')
title({['Error Comparison with Ridge Reg.; N_t_r_a_i_n =',num2str(N_train),...
        '  N_t_e_s_t =',num2str(N_test)];['\lambda =',num2str(lambda)]})
xlabel('M') % x-axis label
ylabel('ERMS') % y-axis label
legend('Training Error','Test Error')

