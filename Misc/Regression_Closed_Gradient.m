% https://www.mathworks.com/help/stats/sample-data-sets.html
close all
clearvars

%Load the carbig dataset
load carbig
%% Manipulating dataset

%Extract columns that we'll work with and Scaling the columns
Z = [((Weight-min(Weight)+1)/(max(Weight)-min(Weight))) ((Horsepower-min(Horsepower)+1)/(max(Horsepower)-min(Horsepower)))];

%Finding NaN values and deleting those rows
[row,col]=find(isnan(Z(:,2)));
a = size(row);
Z(row,:) = [];   
N=size(Z);
%Just visualize the Horsepower w.r.t Weight
figure
plot(Z(:,1),Z(:,2),'bo');

%% Closed form solution
tic
T = Z(:,2); %Target Vector
VarX = [Z(:,1) ones(N(1),1)]; %Variable

A = VarX'*VarX;
A = inv(A);
B = VarX'* T;

W = A*B;

x1=linspace(0,1,400);
y = W(1)*x1 + W(2);
toc
figure
plot(Z(:,1),Z(:,2),'bo');
hold on
plot(x1,y,'r');
hold off
%% Gradient Descent Method
tic
M = 200;
T = Z(:,2);
J = zeros(1,M);
VarX = [Z(:,1) ones(N(1),1)];
W = [0.0 0.0];
rho = 0.001;

for i=1:M   
    Y = VarX*W';
    J(i) = sum((Y - T).^2);
    gradJ = 2*rho*(VarX'*(Y-T));
    W = W-gradJ';  
    plot(i,J(i),'ro');
    hold on
end
hold off
%figure
%plot(1:M,J)
pause(5);

x1=linspace(0,1,400);
y = W(1)*x1 + W(2);
toc
figure
plot(Z(:,1),Z(:,2),'bo');
hold on
plot(x1,y,'r');
hold off

%% Using Built-in fitlm() linear regression model
tic
mdl = fitlm(Z(:,1),Z(:,2),'linear')

x1=linspace(0,1,400);
y = mdl.Coefficients{2,1}*x1 + mdl.Coefficients{1,1};
toc
figure
plot(Z(:,1),Z(:,2),'bo');
hold on
plot(x1,y,'r');
hold off
