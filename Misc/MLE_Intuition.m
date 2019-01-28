
clear
close all

rng(50)
sig=2.0;
mu=5.0;
N=100;
x = mu + sig*randn(N,1);

plot(x,zeros(N,1),'ro')
grid

xx=min(x)-2:0.01:max(x)+2;
p=(1/(sig*sqrt(2*pi)))*exp(-0.5*((xx-mu)/sig).^2);
hold on
h=plot(xx,p,'LineWidth',2);
ylim([-0.1 0.35])
L=prod((1/(sig*sqrt(2*pi)))*exp(-0.5*((x-mu)/sig).^2));
legend(h,['Likelihood=',num2str(L)]);

pause
mu=2.0;
p=(1/(sig*sqrt(2*pi)))*exp(-0.5*((xx-mu)/sig).^2);
L=prod((1/(sig*sqrt(2*pi)))*exp(-0.5*((x-mu)/sig).^2));
h=plot(xx,p,'LineWidth',2,'DisplayName',['Likelihood=',num2str(L)]);

pause
mu=5.0;
sig=1.2;
p=(1/(sig*sqrt(2*pi)))*exp(-0.5*((xx-mu)/sig).^2);
L=prod((1/(sig*sqrt(2*pi)))*exp(-0.5*((x-mu)/sig).^2));
h=plot(xx,p,'LineWidth',2,'DisplayName',['Likelihood=',num2str(L)]);

pause
mu=7.0;
sig=1.5;
p=(1/(sig*sqrt(2*pi)))*exp(-0.5*((xx-mu)/sig).^2);
L=prod((1/(sig*sqrt(2*pi)))*exp(-0.5*((x-mu)/sig).^2));
h=plot(xx,p,'LineWidth',2,'DisplayName',['Likelihood=',num2str(L)]);

