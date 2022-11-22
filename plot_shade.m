% x is the x-axis


function plot_shade(x,y)

yl = y(:,1);
ym = y(:,2);
yu = y(:,3);

patch([x'  fliplr(x')],[yu'  fliplr(yl')],[0.7 0.7 0.7], 'EdgeColor','none');
hold on;
plot(x,ym,'b');
hold off;