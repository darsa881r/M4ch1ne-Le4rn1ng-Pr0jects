close all
clearvars
clc
%% Using resize image to 20 x 20 or 400 pixels

%feature_vector_edge = zeros(2800,400);
feature_vector_gray = zeros(2800,400);

timeStart1=tic;

load('dataset.mat');
%feature_vector_rgb = data.data;

timeStart2=tic;
for i = 1:2800 % the image number to work on
    
    R = data.data(i,1:6400);
    r = reshape(R,[80, 80]);

    G = data.data(i,6401:12800);
    g = reshape(G,[80, 80]);

    B = data.data(i,12801:19200);
    b = reshape(B,[80, 80]);

%   I = cat(3, r,g,b);
    I = (r+g+b)./3; % creating grayscale image
    I1 = imresize(I,0.25);
    
%   BW = edge(I1,'canny',0.30);
%   BW_D = double(BW);
%   imshow(BW_D,[]);
    
%    feature_vector_edge(i,1:400) = BW_D(:)';
    feature_vector_gray(i,1:400)= I1(:)';
end

clear BW BW_D r g b R G B I I1

timeElapsed2=toc(timeStart2);
%disp(['image pre-processing time: ',num2str(timeElapsed2)])


%% K-fold CrossValidation

p = randperm(2800,2800)';
data_total = feature_vector_gray(p,:); %change here to use feature_vector_edge / feature_vector_gray / feature_vector_rgb
labels = data.labels(p,:);

K = 5;
N_total = 2800;
for i = 1:K
    L{i} = crossvalind('Kfold',(N_total/K),N_total);
end

N_test = (N_total/K);
rho = 0.1; %learning rate, rho = 1 gives best result for grayscale;rho=0.01 for edge
lambda = 0.0; %regulariztion parameter lambda =0.1 default for grayscale/edge
iter = 1000; % for gradient_descent; can be more or less depends on convergence 2000 for edge; 1000 for gray

timeElapsed1=toc(timeStart1);
%disp(['Pre-training time: ',num2str(timeElapsed1)])

%q=1;

%%
for q = 1:K
        timeStart3(q)=tic;

        data_test = data_total(L{q},1:400)./255;   % when necessary take only 50 dimension and normalizing using 255 when not BnW
        label_test = labels(L{q},1);
        
        data_train = data_total(:,1:400)./255;   %  when necessary take only 50 dimension and normalizing using 255 when not BnW
        label_train = labels;
        data_train(L{q},:)=[];
        label_train(L{q},:)=[];

        % Data Pre-Processing
        [~, dim] = size(data_train);
        X = data_train(:,1:dim); 
        y = label_train;
               
        %%% Generate Design Matrix with Radial basis function  
        [m, n] = size(X);
       
        %for RBF
        s = 1; % for gray use s=1 and for edge use s=0.1
        Phi = zeros(m,m-1);
        count = 0;
        timeStart4(q)=tic;
%         for i = 1:m
%             for j = 1:m-1
%                 Phi(i,j) = exp(-0.5*(s^2)*sum(((X(i,:)-X(j,:)).^2),2));                  
%             end
%             %count=1+count
%         end
        Phi = [ones(m,1) X];
        
        timeElapsed4(q)=toc(timeStart4(q));       
%        disp(['RBF generation time for K=',num2str(q),' : ',num2str(timeElapsed4(q))])         
             
        % Compute Cost and Gradient 
        
        [m1, n1] = size(Phi);
        theta = zeros(n1, 1);
        N_train = length(y); % = m no of rows
  
        costJ = zeros(iter, 1); 
        gradientJ = zeros(size(theta));
        timeStart5(q)=tic;
        for i = 1:iter
            hyp = 1./(1 + exp(-Phi*theta));
            costJ(i) = (-1/N_train) * sum( y .* log(hyp) + (1 - y) .* log(1 - hyp))...
                    + lambda/(2*N_train) * sum( theta(2:end).^2 );
            gradientJ = (rho * (1/N_train)) * (((hyp - y)' * Phi)+(lambda*theta)'); 
            theta = theta - gradientJ'; 
            %i
        end
        timeElapsed5(q)=toc(timeStart5(q));
%        disp(['Cost mimization time for K=',num2str(q),' : ',num2str(timeElapsed5(q))])
        
        hyp_train(:,q)=hyp;
        k(:,q) = (hyp >= 0.5); %hyp is the corresponding probability of y = 1 being true %using 0.5 as threshold but can be changed depending on the accuracy
        N_mis_train = 2240-sum(double(k(:,q) == y));
        tr_acc(q) = mean(double(k(:,q) == y)) * 100;
        tr_err(q) = 100.0-tr_acc(q);
        timeElapsed3(q)=toc(timeStart3(q));       
%        disp(['Training time for K=',num2str(q),' : ',num2str(timeElapsed3(q))])
        
        %Plot the convergence graph        
        figure(1);
        plot(1:numel(costJ), costJ, 'LineWidth', 2);
        hold on
        xlabel('Number of iterations');
        ylabel('Cost J');
        warning('off')
        legend('K=1','K=2','K=3','K=4','K=5');
%% Prediction

        timeStart6(q)=tic;
        
        X_test = data_test; 
        y_true = label_test;
        
        [mt, nt] = size(X_test);
    
        % Design Matrix with 1 appended at first
        Phi_test = zeros(mt,m-1);

%         for i = 1:mt
%             for j = 1:m-1
%                 Phi_test(i,j) = exp(-0.5*(s^2)*sum(((X_test(i,:)-X(j,:)).^2),2));        
%             end
%         end
        Phi_test = [ones(mt,1) X_test];

        hyp_test(:,q) = 1./(1 + exp(-Phi_test*theta));

        k_test = (hyp_test >= 0.5); %using 0.5 as threshold but can be changed depending on the accuracy
        N_mis_test =  560-sum(double(k_test(:,q) == y_true));
        te_acc(q) = mean(double(k_test(:,q) == y_true)) * 100;
        te_err(q) = 100.0-te_acc(q);
        
        timeElapsed6(q)=toc(timeStart6(q));
%        disp(['Testing time for K=',num2str(q),' : ',num2str(timeElapsed3(q))])

        figure(2);
        plot((1:numel(tr_err)),tr_err );
        hold on
        plot((1:numel(te_err)),te_err );
        hold off
        title('Test Error vs. Train Error');
        xlabel('Number of K-fold');
        ylabel('Error Percentage');
        legend('train error','test error');

        axis([1 K 0 30]);
    
end
warning('on')
fprintf('Average Number of misclassifications in train set (out of 2240): %f\n', mean(N_mis_train));
avg_tr_acc = mean(tr_acc);
avg_tr_err = mean(tr_err);
fprintf('Average Train Error (percent): %f\n', avg_tr_err);

fprintf('Average Number of misclassifications in test set (out of 560): %f\n', mean(N_mis_test));
avg_te_acc = mean(te_acc);
avg_te_err = mean(te_err);
fprintf('Average Test Error (percent): %f\n', avg_te_err);

total_train_time = timeElapsed1 + mean(timeElapsed3);
total_test_time = mean(timeElapsed6);

disp(['Average Training time : ',num2str(total_train_time)])
disp(['Average Test time : ',num2str(total_test_time)])
