%% Logistic Regression

%% Initialization
clearvars ; 
close all; 
clc

%% Generating Synthetic Data with slight overlap
tic
rng(200,'v4');

class1=[mvnrnd([1 3],[1 0; 0 1],100) ones(100,1)];
class2=[mvnrnd([4 1],[1 0; 0 1],100) zeros(100,1)];

data_ordered = cat(1, class1, class2);
p = randperm(200,200)';

data = data_ordered(p,:);

X = data(:, [1, 2]); 
y = data(:, 3);    

%fprintf(['Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n']);

index_yes = find(y == 1);
index_no = find(y == 0);

figure; 
hold on;
plot(X(index_yes, 1), X(index_yes, 2),...
	'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(index_no, 1), X(index_no, 2),...
	'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

hold off;

% Labels and Legend
title('Synthetic Data')
xlabel('x1')
ylabel('x2')

% Specified in plot order
legend('class01', 'class02')

%fprintf('\nProgram paused. Press enter to continue.\n');
%pause;
disp('Data pre-processing time')
toc

%% Compute Design Matrix, Cost and Gradient 
tic
[m, n] = size(X);

% Design Matrix with 1 appended at first
X1 = [ones(m, 1) X];

% Initialize
theta = zeros(n + 1, 1);
N_train = length(y); % = m
iter = 200;
rho = 0.1;
costJ = zeros(iter, 1); 
gradientJ = zeros(size(theta));
%%
for i = 1:iter
    
    hyp = 1./(1 + exp(-X1*theta))
    
    %Normalizing the cot fucntion with N_train
    costJ(i) =   (-1/N_train)* sum( y .* log(hyp) + (1 - y) .* log(1 - hyp) );
    
    %finding the gradient
    gradientJ = (rho * (1/N_train)) * ((hyp - y)' * X1); 
    theta = theta - gradientJ'; 
end

% Plot the convergence graph
figure;
plot(1:numel(costJ), costJ, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% % Display gradient descent's result
% fprintf('Theta computed from gradient descent: \n');
% fprintf(' %f \n', theta);
% fprintf('\n');

% fprintf('\nProgram paused. Press enter to continue.\n');
% pause;
%% Plot Decision Boundary

figure; 
hold on;
plot(X1(index_yes, 1+1), X1(index_yes, 2+1),...
	'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X1(index_no, 1+1), X1(index_no, 2+1),...
	'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

% Labels and Legend
title('Synthetic Data')
xlabel('x1')
ylabel('x2')
legend('class01', 'class02')

%plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
plot_x = X(:,2);

% Calculate the decision boundary line
plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

% Plot, and adjust axes for better viewing
plot(plot_x, plot_y)

% Legend, specific for the exercise
legend('class01', 'class02', 'Decision Boundary')



% fprintf('\nProgram paused. Press enter to continue.\n');
% pause;

%% Prediction and Accuracy

% prob = 1./(1 + exp(-([1 3.5 1.5] * theta)));
% fprintf('For a co-ordinate 1.5 and 3.5, we predict an positive probability of %f\n\n', prob);

% Compute accuracy on our training set

m = size(X, 1); % Number of training examples

k = (hyp >= 0.5);

fprintf('Train Accuracy: %f\n', mean(double(k == y)) * 100);

disp('Data processing and Visualizing time')
toc
