close all
clearvars

%Load the carbig dataset
load carbig

%% Manipulating dataset
%Finding NaN values and deleting those rows
a=isnan(Horsepower);
Horsepower(a,:) = []; 
Weight(a,:) = []; 

%Extract columns that we'll work with and Scaling the columns
Z = [Weight/max(Weight) Horsepower/max(Horsepower)]; 
N=size(Z);

%% Closed form solution
tic
T = Z(:,2); %Target Vector
VarX = [Z(:,1) ones(N(1),1)]; %Variable

A = VarX'*VarX;
A = inv(A);
B = VarX'* T;

W1 = A*B;
toc

%% Gradient Descent Method
tic
M = 500;
J = zeros(1,M);
T = Z(:,2);
VarX = [Z(:,1) ones(N(1),1)];
W2 = [0.0 0.0];
rho = 0.001;

for i=1:M   
    Y = VarX*W2';
    J(i) = sum((Y - T).^2);
    gradJ = 2*rho*(VarX'*(Y-T));
    W2 = W2-gradJ';  
    plot(i,J(i),'bo');
    hold on
end
hold off
title('Criterion Function ')
xlabel('Iterations') % x-axis label
ylabel('Cost function,J(W)') % y-axis label
toc

%% Visualization

figure
subplot(1,2,1)
x1=linspace(1500,5500,400);
y1 = W1(1)*x1*(max(Horsepower)/max(Weight)) + W1(2)*max(Horsepower);
plot(Weight,Horsepower,'rx');
hold on
plot(x1,y1,'b');
hold off
title('Matlab carbig dataset ')
xlabel('Weights') % x-axis label
ylabel('Horsepower') % y-axis label
legend('Datapoints','Closed Form')

subplot(1,2,2)
x2=linspace(1500,5500,400);
y2 = W2(1)*x1*(max(Horsepower)/max(Weight)) + W2(2)*max(Horsepower);
plot(Weight,Horsepower,'rx');
hold on
plot(x2,y2,'g');
hold off
title('Matlab carbig dataset ')
xlabel('Weights') % x-axis label
ylabel('Horsepower') % y-axis label
legend('Datapoints','Gradient Descent')

