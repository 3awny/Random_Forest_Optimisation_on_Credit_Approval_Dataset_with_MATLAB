function [accuracy, fscore, gmean] = performance(confusion)
    % This function will calculate the performance of the model by
    % calculating the accuracy, fscore and gmean
    
    accuracy = (confusion(1,1)+confusion(2,2))/sum(confusion, 'all');   % calculating accuracy
    precision = confusion(1,1)/(confusion(1,1)+confusion(2,1)); % calculating precision
    recall = confusion(1,1)/(confusion(1,1)+confusion(1,2));    % calculating recall
    gmean = sqrt(recall*precision); % calculating gmean
    fscore = 2*(precision*recall/(precision+recall)); % calculating fscore
    
end