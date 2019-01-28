clearvars ; 
close all; 
clc

data_X = [ 0 0;
           1 0;
           0 1;
           1 1; ];
[N,~]=size(data_X); 
data_T = [0;1;1;0;];
scatter(data_X(data_T(:,1)==0,1),data_X(data_T(:,1)==0,2),'bx'); 
scatter(data_X(data_T(:,1)==1,1),data_X(data_T(:,1)==1,2),'ro');
hold off

% Initialization

N_layer = 3;                %including input and output layers
N_unit = zeros(N_layer,1);
N_unit(1)= 2;               %number of input dimensions
N_unit(N_layer)= 1;         %number of output units
N_unit(2)= 2;   %assuming same number of units in the hidden layers

%initializing the weight and biases
% for j=1:1
%     rng(18)
% 
%     W1 = randn(N_unit(2),N_unit(1));
%     B1 = repmat(0.1*randn(N_unit(2),1),1,N);
% 
%     W2 = randn(N_unit(3),N_unit(2));
%     B2 = repmat(0.1*randn(N_unit(3),1),1,N);

    W1 =[0.2068    0.8243;
        0.0155   -1.6221;];

    B1 =[0.0712    0.0712    0.0712    0.0712;
        0.0168    0.0168    0.0168    0.0168;];

    W2 =[ 0.4699    1.6196];

    B2 =[   -0.1123   -0.1123   -0.1123   -0.1123];

    Z1(1:N_unit(1),1:N,1) = data_X';

    rho = 0.05;

    for i=1:20000

        % Forward Pass 

        A1=W1*Z1+B1;
        Z2=max(0,A1); % performing ReLu    
        A2=W2*Z2+B2;
        Y1=1./(1 + exp(-A2)); % performing sigmoid 

        % Back Propagation

        delta3 =(Y1-data_T');
        delta2=((W2'*delta3)).*repmat(double(A2>0),2,1);

        delta3_bar = sum(delta3,2);
        delta2_bar = sum(delta2,2);

        % Update Equation

        W2=W2-rho*delta3*Z2';
        W1=W1-rho*delta2*Z1';

        B2= B2-rho*delta3_bar;
        B1= B1-rho*delta2_bar;

    end

predicted_Y = (Y1>0.50)

[X,Y] = meshgrid(-1.5:0.05:1.5,-1.5:0.05:1.5);
Xnew = [X(:) Y(:)];
B1 =repmat([0.0712;0.0168;],1,length(Xnew));
%B1 =repmat([-0.5;0.1;],1,length(Xnew));

Z1 = W1*Xnew'+B1;
A1 = max(0,Z1);
B2 =repmat(-16.0491,1,length(Xnew)); %16.0491

Z2 = W2*A1+B2;
A2 = 1./(1 + exp(-Z2));

var3 = (A2'>0.5);

y = reshape(double(var3),size(X));
surf(X,Y,double(y))
hold on
scatter(data_X(data_T(:,1)==0,1),data_X(data_T(:,1)==0,2),'bx'); 
scatter(data_X(data_T(:,1)==1,1),data_X(data_T(:,1)==1,2),'ro');
hold off

%% Linear Regression with 3 hidden units

clearvars


rng('default')
rng(100);
X = (2*rand(1,50)-1)';

T = (sin(2*pi*X')+0.3*randn(1,50))';


N_in = size(X,2);
N_HL= 1;
N_HUnits = 3;
N_Output_unit = 1;

rho = 0.003; 
max_iters = 40000;
N = size(X,1);


% initalizing weights and biases

rng('default')
rng(0);

W1 = randn(N_in,N_HUnits); %/sqrt(numberOfInputUnits);

rng(0);

W2 = randn(N_HUnits,N_Output_unit); %/sqrt(N_Output_unit);

% Initialize the bias 

B1 = zeros(1,N_HUnits)+0.01;
B2 = zeros(1,N_Output_unit)+0.01;

for i=1:max_iters
    Z1 = X*W1+B1;
    A1 = tanh(Z1);
    Z2 = A1*W2+B2;
    A2 = Z2;
    
    delta3 = A2 - T;
    deltaW2 = (A1')*(delta3);
    deltaB2 = sum(delta3,1);
    delta2 = (delta3*(W2')) .* (1-A1.^2);
    deltaW1 = (X')*delta2;
    deltaB1 = sum(delta2,1);

    W1 = W1-rho*deltaW1;
    W2 = W2-rho*deltaW2;
    B1 = B1-rho*deltaB1;
    B2 = B2-rho*deltaB2;
 
    loss = sqrt(sum((A2-T).^2)/size(T,1));
end

figure
plot(X,T,'rx')
hold on
[~,dx] = sort(X(:,1)); % sort just the first column
sortedmat = [X(dx,:) A2(dx,:)]; 
plot(sortedmat(:,1),sortedmat(:,2),'g-')
hold off

%% Linear Regression with 20 hidden units

clearvars


rng('default')
rng(100);
X = (2*rand(1,50)-1)';

T = (sin(2*pi*X')+0.3*randn(1,50))';


N_in = size(X,2);
N_HL= 1;
N_HUnits = 20;
N_Output_unit = 1;

rho = 0.003; 
max_iters = 40000;
N = size(X,1);


% initalizing weights and biases

rng('default')
rng(0);

W1 = randn(N_in,N_HUnits); %/sqrt(numberOfInputUnits);

rng(0);

W2 = randn(N_HUnits,N_Output_unit); %/sqrt(N_Output_unit);

% Initialize the bias 

B1 = zeros(1,N_HUnits)+0.01;
B2 = zeros(1,N_Output_unit)+0.01;

for i=1:max_iters
    Z1 = X*W1+B1;
    A1 = tanh(Z1);
    Z2 = A1*W2+B2;
    A2 = Z2;
    
    delta3 = A2 - T;
    deltaW2 = (A1')*(delta3);
    deltaB2 = sum(delta3,1);
    delta2 = (delta3*(W2')) .* (1-A1.^2);
    deltaW1 = (X')*delta2;
    deltaB1 = sum(delta2,1);

    W1 = W1-rho*deltaW1;
    W2 = W2-rho*deltaW2;
    B1 = B1-rho*deltaB1;
    B2 = B2-rho*deltaB2;
 
    loss = sqrt(sum((A2-T).^2)/size(T,1));
end

figure
plot(X,T,'rx')
hold on
[~,dx] = sort(X(:,1)); % sort just the first column
sortedmat = [X(dx,:) A2(dx,:)]; 
plot(sortedmat(:,1),sortedmat(:,2),'b-')
hold off