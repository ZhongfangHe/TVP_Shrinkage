

function plot_TVP(x)


K = length(x);
m = K/2;
if m == round(m)
    for j = 1:K
        subplot(m,2,j);
        plot(prctile(x{j},[5 50 95])');
%         plot_prctile(x{j});
        title(['para ',num2str(j)]);
    end
else
    m = (K+1)/2;
    for j = 1:K
        subplot(m,2,j);
        plot(prctile(x{j},[5 50 95])');
%         plot_prctile(x{j});
        title(['para ',num2str(j)]);
    end
end
