
clear
close all

N=500;
x=randn(N,1)+0.5;  %try randn()

%closed form of MLE of Gaussian mean
m_closed=(1/500)*sum(x);
figure
hold on
grid
xlabel('n'); ylabel('\mu');

ini=rand(1,1);

%Gradient descent form of MLE of Gaussian mean
m_GD_o=ini;
err=1;
rho=0.001;
iter=0;
while err > 1e-6
    m_GD=m_GD_o+rho*sum(x-m_GD_o);
    err=abs(m_GD-m_GD_o);
    m_GD_o=m_GD;
    iter=iter+1;
end
title(['Grad. Descent Mean = ',num2str(m_GD,'%.3f'),' reached in ',...
    num2str(iter),' iterations']);

%Stochastic Gradient descent form of MLE of Gaussian mean - samples
%presented in random order
mo=ini;
for n=1:N
    mn=mo+((1/n)*(x(n)-mo)); 
    plot(n,mo,'ro'); 
    mo=mn; 
end

%Stochastic Gradient descent form of MLE of Gaussian mean - samples
%presented in a different random order
x=x(randperm(N));
mo=ini;
for n=1:N
    mn=mo+((1/n)*(x(n)-mo)); 
    plot(n,mo,'gx'); 
    mo=mn; 
end

%Stochastic Gradient descent form of MLE of Gaussian mean - samples
%presented in a different random order
x=x(randperm(N));
mo=ini;
for n=1:N
    mn=mo+((1/n)*(x(n)-mo)); 
    plot(n,mo,'m*'); 
    mo=mn; 
end

%Stochastic Gradient descent form of MLE of Gaussian mean - samples
%sorted in ascending order
%Trajectory may depend on shape of the data-generating distribution
x=sort(x);
mo=ini;
for n=1:N
    mn=mo+((1/n)*(x(n)-mo)); 
    plot(n,mo,'bs'); 
    mo=mn; 
end

%Stochastic Gradient descent form of MLE of Gaussian mean - samples
%sorted in descending order
%Trajectory may depend on shape of the data-generating distribution
x=flip(x);
mo=ini;
for n=1:N
    mn=mo+((1/n)*(x(n)-mo)); 
    plot(n,mo,'cs'); 
    mo=mn; 
end

plot(1:N,m_closed*ones(N,1),'LineWidth',2)
