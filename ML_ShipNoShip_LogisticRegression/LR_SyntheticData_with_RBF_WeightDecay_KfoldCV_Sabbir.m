%% Logistic Regression

clearvars ; 
close all; 
clc
%% Generating Synthetic Data with slight overlap

rng(200,'v4');

class1=[mvnrnd([1 3],[1 0; 0 1],100) ones(100,1)];
class2=[mvnrnd([4 1],[1 0; 0 1],100) zeros(100,1)];

data_ordered = cat(1, class1, class2);
p = randperm(200,200)';
data = data_ordered(p,:);

%% K-fold CrossValidation

K = 5;
N_total = 200;
for i = 1:K
I{i} = crossvalind('Kfold',(N_total/K),N_total);
end

N_test = (N_total/K);

for q = 1:K
    
        data_test = data(I{q},:);
        data_train = data;
        data_train(I{q},:)=[];

        % Data Pre-Processing

        X = data_train(:,1:2); 
        y = data_train(:,3);    
        index_yes = find(y == 1);
        index_no = find(y == 0);

        % Generate Design Matrix with Radial basis function  
        [m, n] = size(X);
        s = 1.0;
        % Design Matrix with 1 appended at first
        Phi = zeros(m,m-1);

        for i = 1:m
            for j = 1:m-1
                Phi(i,j) = exp(-0.5*(s^2)*sum(((X(i,:)-X(j,:)).^2),2));        
            end
        end
        Phi = [ones(m,1) Phi];

        % Compute Cost and Gradient 

        [m1, n1] = size(Phi);
        theta = zeros(n1, 1);
        N_train = length(y); % = m no of rows
        iter = N_train;
        rho = 0.1;
        costJ = zeros(iter, 1); 
        gradientJ = zeros(size(theta));
        lambda = 0;

        for i = 1:iter

            hyp = 1./(1 + exp(-Phi*theta));
            costJ(i) = (-1/N_train) * sum( y .* log(hyp) + (1 - y) .* log(1 - hyp))...
                    + lambda/(2*N_train) * sum( theta(2:end).^2 );
            gradientJ = (rho * (1/N_train)) * (((hyp - y)' * Phi)+(lambda*theta)'); 
            theta = theta - gradientJ'; 
        end

        %Plot the convergence graph
        figure(1);
        plot(1:numel(costJ), costJ, 'LineWidth', 2);
        hold on
        xlabel('Number of iterations');
        ylabel('Cost J');
        warning('off')
        legend('K=1','K=2','K=3','K=4','K=5'); 
        % Prediction
        hyp_train(:,q)=hyp;
      
        X_test = data_test(:,1:2); 
        y_true = data_test(:,3);
        
        [mt, nt] = size(X_test);
        s = 1;
        % Design Matrix with 1 appended at first
        Phi_test = zeros(mt,m-1);

        for i = 1:mt
            for j = 1:m-1
                Phi_test(i,j) = exp(-0.5*(s^2)*sum(((X_test(i,:)-X(j,:)).^2),2));        
            end
        end

        Phi_test = [ones(mt,1) Phi_test];
        hyp_test(:,q) = 1./(1 + exp(-Phi_test*theta));

        k(:,q) = (hyp >= 0.5); %hyp is the corresponding probability of y = 1 being true
        
        tr_acc(q) = mean(double(k(:,q) == y)) * 100;
        tr_err(q) = 100.0-tr_acc(q);
%         fprintf('Train Accuracy: %f\n', acc(q));
%         fprintf('Train Error: %f\n', err(q));
        k_test = (hyp_test >= 0.5);
        te_acc(q) = mean(double(k_test(:,q) == y_true)) * 100;
        te_err(q) = 100.0-te_acc(q);
        figure(2);
        plot((1:numel(tr_err)),tr_err );
        hold on
        plot((1:numel(te_err)),te_err );
        hold off
        title('Test Error vs. Train Error');
        legend('train error','test error');
        axis([1 K 0 30]);

end
warning('on')
avg_tr_acc = mean(tr_acc);
avg_tr_err = mean(tr_err);
fprintf('Average Train Error: %f\n', avg_tr_err);

avg_te_acc = mean(te_acc);
avg_te_err = mean(te_err);
fprintf('Average Test Error: %f\n', avg_te_err);
