clear
close all

rng(200)
N=10;
Ntest=N;
x=2*rand(N,1)-1;
y=-x.^2+(0.1)*randn(N,1);
xtest=2*rand(Ntest,1)-1;
ytest=-xtest.^2+(0.1)*randn(Ntest,1);

X=[x.^9 x.^8 x.^7 x.^6 x.^5 x.^4 x.^3 x.^2 x ones(length(x),1)];
Xtest=[xtest.^9 xtest.^8 xtest.^7 xtest.^6 xtest.^5 xtest.^4 ...
    xtest.^3 xtest.^2 xtest ones(length(xtest),1)];
for lambda=[0 0.5 5 10]
    theta=(X'*X+lambda*eye(10,10))\(X'*y);
    TrainErr=norm(y-X*theta)^2;
    TestErr=norm(ytest-Xtest*theta)^2;
    figure
    plot(x,y,'rx','Markersize',10,'LineWidth',5)
    xlim([-1 1]);
    xlabel('x'); ylabel('y');
    hold on
    plot(xtest,ytest,'gs','Markersize',10,'LineWidth',2)
    fplot(@(t) -t.^2,[-1 1],'g--','LineWidth',2);
    fplot(@(t) theta(1)*t.^9+theta(2)*t.^8+theta(3)*t.^7+theta(4)*t.^6+...
        theta(5)*t.^5+theta(6)*t.^4+theta(7)*t.^3+theta(8)*t.^2+...
        theta(9)*t+theta(10),[-1 1],'b','LineWidth',2)
    legend(['Train Error = ',num2str(TrainErr,'%.2f')],...
        ['Test Error = ',num2str(TestErr,'%.2f')],...
        'True Model','Predicted Model','Location','SE');
    title(['\lambda=',num2str(lambda),'      |\theta|=',...
        num2str(norm(theta))]);
    grid
end
