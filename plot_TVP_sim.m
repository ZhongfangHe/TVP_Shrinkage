%x = draws.beta (cell of posterior draws),
%btrue (n-by-K matrix of true values)

function plot_TVP_sim(x,btrue)


K = length(x);
Kb = size(btrue,2);
if K~= Kb
    error('mismatch dimension');
end
m = K/2;
if m == round(m)
    for j = 1:K
        subplot(m,2,j);
%         plot(prctile(x{j},[5 50 95])');
%         hold on;
%         plot(btrue(:,j),'k');
%         hold off;
% %         plot_prctile(x{j});
%         title(['para ',num2str(j)]);        

        para_est = prctile(x{j},[5 50 95])';
        para_true = btrue(:,j);
        n = length(para_true);
        plot_shade((1:n)', para_est);
        hold on;
        plot((1:n)', para_true,'r--');
        hold off;
        title(['para ',num2str(j)]);     
        
    end
else
    m = (K+1)/2;
    for j = 1:K
        subplot(m,2,j);
%         plot(prctile(x{j},[5 50 95])');
%         hold on;
%         plot(btrue(:,j),'k');
%         hold off;        
% %         plot_prctile(x{j});
%         title(['para ',num2str(j)]);
        
        para_est = prctile(x{j},[5 50 95])';
        para_true = btrue(:,j);
        n = length(para_true);
        plot_shade((1:n)', para_est);
        hold on;
        plot((1:n)', para_true,'r--');
        hold off;
        title(['para ',num2str(j)]);         
    end
end
