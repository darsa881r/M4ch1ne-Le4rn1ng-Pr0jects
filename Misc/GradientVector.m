%
close all
clear

[x1,x2] = meshgrid(-2:.1:2, -2:.1:2);
z = x1 .* exp(-x1.^2 - x2.^2);
surface(x1,x2,z)

figure
contour(x1,x2,z);
hold on
grad_x1=exp(-x1.^2 - x2.^2) - 2*x1.^2 .* exp(-x1.^2 - x2.^2);
grad_x2=-2.*x1.*x2 .* exp(-x1.^2 - x2.^2);
quiver(x1,x2,grad_x1,grad_x2);