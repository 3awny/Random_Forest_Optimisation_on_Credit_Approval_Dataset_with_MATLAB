%% 1. Splitting the Data to Test and Train -----------------------------------

% GRID SEARCH IN SECTION 5 OF THIS SCRIPT

y = readmatrix('labels.csv');
x = readmatrix('no_label_credit.csv');
y = y';

% Train Test Split
rng(1)
cv = cvpartition(size(x,1),'HoldOut',0.3);
idx = cv.test;

dataTrain_x = x(~idx,:);
dataTest_x  = x(idx,:);

cv2 = cvpartition(y,'HoldOut',0.3);
idx2 = cv2.test;

dataTrain_y = y(~idx2,:);
dataTest_y  = y(idx2,:);
icol = size(dataTrain_x,2);


%% 2. OPTIMISE MIN-LEAF Parameter -----------------------------------------------


leaf = [5 10 20 50 100];
colour = 'rbcmy';
figure
% MIN LEAF SIZE OPTIMISATION
for i=1:length(leaf)
    rng(1)
    mdl1 = TreeBagger(300,dataTrain_x,dataTrain_y,'Method','classification',...
        'OOBPred','On', 'MinLeafSize', leaf(i));
    %errors = [errors oobError(mdl)];
    oobError(mdl1)
    plot(oobError(mdl1),colour(i))
    hold on
end
xlabel 'Number of Grown Trees'
ylabel 'Mean Squared Error'
title('Random Forest Min Leaf Size Optimisation')
legend({'5','10','20','50','100'},'Location', 'NorthEast')
hold off


%% 3. INITIAL MODEL TRAINING WITH OPTIMISED MIN-LEAF SIZE (10) and 300 Trees --------------------------------

% Building an optimised min-leaf size model with an arbitrary number of trees to 
% compare with grid search optimised model performance.
% Min-leaf size was not included in the hyperparamater grid search (section 5) due to time contraints


min_leaf = 10; % optimal value

initial_model = TreeBagger(300,dataTrain_x,dataTrain_y,...
    'OOBPred','On','Method','classification', 'MinLeafSize', min_leaf);

[label_init, scores_initial] = predict(initial_model, dataTest_x);
A_init = cell2mat(label_init);
Z_init = str2num(A_init);
figure
cm_initial = confusionmat(dataTest_y, Z_init, 'Order', [1 0]);
figure
cm_chart = confusionchart(cm_initial)
title('Confusion Matrix for initial RF Model')

[accur_init, f1score_init, gmean_final_init] = performance(cm_initial);
intial_f1score = f1score_init
inital_accuracy = accur_init

% Out of Bag classification Error
oobErrorEnsembleBagged = oobError(initial_model);
fig = figure;
plot(oobErrorEnsembleBagged)
xlabel 'Number of Grown Trees';
ylabel 'Out-of-bag classification error';
print(fig, '-dpdf', date);

oobPredict(initial_model)

% show trees
view(initial_model.Trees{1})
view(initial_model.Trees{1}, 'mode', 'graph')



%% 4. feature importance for initial model with optimised min_leaf ---------------------------------


mdl = TreeBagger(300, dataTrain_x, dataTrain_y, 'method',...
    'classification', 'OOBVarImp','On', 'MinLeafSize',min_leaf);

figure
plot(oobError(mdl))
xlabel 'NO. of Grown Trees'
ylabel 'Out-of-bag Mean Squared Error'

figure
bar(mdl.OOBPermutedVarDeltaError)
xlabel 'Feature Number'
ylabel 'Out-of-Bag Feature Importance'
idxvar = find(mdl.OOBPermutedVarDeltaError>0.7);


%% 5. Running the GridSearch on NO. of trees and NO. of features with cross-validation ----------------------------------


% Initialising Parameters
trees = [2 5 8 16 32 64 128 170 200:25:300 320:20:400 500];
features = [1:size(dataTrain_x,2)];

AUC_crossval_Grid = zeros(length(trees),length(features));
F1_crossval_Grid = zeros(length(trees),length(features));
AUC_FPTP_Grid = zeros(length(trees),length(features));


% To store results 
trainingError = [];
validationError = [];
oobErr = [];
timeTaken = [];

trainingError_mean = zeros(length(features),length(trees));
validationError_mean = zeros(length(features),length(trees));
oobErr_mean = zeros(length(features),length(trees));
timeTaken_mean = zeros(length(features),length(trees));

% Storing Results
parameters = [];
errors = [];
final = table;
accuracy = [];
TreeErrorsTable = [];
f1_score = [];

errors_mean = zeros(length(features),length(trees));
accuracy_mean = zeros(length(features),length(trees));

recall = [];
precision = [];

x_cv_check = 0;
y_cv_check = 0;

x_fp_check = 0;
y_tp_check = 0;

mean_errs = [];
mean_acc = [];

tic;
for f=features
    for t=trees
        
        folds = 10;
        indx = crossvalind('kfold', dataTrain_y,folds);
        final_predictions = [];
        final_scores = [];
        y_target = [];

        parameters = [parameters; f; t];
        modelErrorCheck = 100;
        
        for i=1:folds
            x2_fold = dataTrain_x(indx == i,:);
            x1_fold = dataTrain_x(indx ~= i,:);
            y_fold = dataTrain_y(indx ~= i,:);
            y2_fold = dataTrain_y(indx == i,:);
            idx_test = (indx == i);
            %"Model" + i
            rng(1)
            iter_model = TreeBagger(t, x1_fold, y_fold,...
                'NumPredictorsToSample',f,'MinLeafSize',10,...
                'OOBPrediction', 'on', 'Method','classification');
                        
            model_error = oobError(iter_model, 'Mode', 'Ensemble');
            
            if model_error < modelErrorCheck
                modelErrorCheck = model_error;
                tree_error = oobError(iter_model);
            end
            
            errors(i) = model_error;              
            [predicted_labels, scores] = predict(iter_model, x2_fold);
            
            final_predictions = [final_predictions; predicted_labels];
            final_scores = [final_scores; scores];
            y_target = [y_target; dataTrain_y(idx_test)];
            y2_fold2 = num2cell( num2str(y2_fold) );
            model_cm = confusionmat(y2_fold2, predicted_labels);
            model_acc = 100*sum(diag(model_cm))./sum(model_cm(:));
            accuracy(i) = model_acc;
            [accuracyt,f_scoret,gmeant] = performance(model_cm);
            f1_score(i)=f_scoret;
            
            trainingError(i) = mean(error(iter_model, x1_fold, y_fold));
            validationError(i) = mean(error(iter_model, x2_fold, y2_fold));
            oobErr(i) = mean(oobError(iter_model));
            timeTaken(i) = toc/60;
            
        end
        
        % attempt to plot highest avg ROC Curve
        [x_fp, y_tp, t_cv, auc_conf] = perfcurve(y_target,final_scores(:,2),'1');
        if mean(x_fp)>x_fp_check && mean(y_tp)>y_tp_check
            x_fp_check = mean(x_fp);
            y_tp_check = mean(y_tp);
            figure
            plot(x_fp,y_tp)
            title('Random Forest ROC Curve - Highest Average')
            xlabel('False positive rate')
            ylabel('True positive rate')
            legend(sprintf('%s_%d_%s_%d','NO. of Features:',f,...
                'and NO. of Trees:',t));
        end
            
        
        [x_cv, y_cv, t_cv, auc_cv] = perfcurve(y_target, final_scores(:,2),...
            '1', 'xCrit', 'reca', 'yCrit', 'prec');
        % Attempt to plot Best Precision-Recall Curve from Grid Search
        if mean(x_cv)>x_cv_check && mean(y_cv) >y_cv_check
            x_cv_check = mean(x_cv);
            y_cv_check = mean(y_cv);
            figure
            plot(x_cv, y_cv)
            title('Best Precision-Recall curve')
            xlabel('Recall'); ylabel('Precision');
            legend(sprintf('%d_%d',f,t));
        % To plot best Recall Curve
        elseif mean(x_cv)>x_cv_check
            x_cv_check = mean(x_cv);
            y_cv_check = mean(y_cv);
            figure
            plot(x_cv, y_cv)
            title('Best Recall curve')
            xlabel('Recall'); ylabel('Precision');
            legend(sprintf('%d_%d',f,t));
        % To plot best Precision Curve
        elseif mean(y_cv)>y_cv_check
            x_cv_check = mean(x_cv);
            y_cv_check = mean(y_cv);
            figure
            plot(x_cv, y_cv)
            title('Best Precision curve')
            xlabel('Recall'); ylabel('Precision');
            legend(sprintf('%d_%d',f,t));
        end
        
        % GRIDS FOR HYPER-PARAMETER STORAGE 
        F1_crossval_Grid(find(features==f),find(trees==t))=mean(f1_score);
        AUC_crossval_Grid(find(features==f), find(trees==t)) = auc_cv;
        AUC_FPTP_Grid(find(features==f),find(trees==t)) = auc_conf;      
        
        TreeErrorsTable = [TreeErrorsTable; tree_error];
        accuracy_mean(find(features==f),find(trees==t))=mean(accuracy);
        errors_mean(find(features==f), find(trees==t))=mean(errors);
        % Attempt to store parameters combination with accuracy and mean in
        % table
        mean_errs = [mean_errs mean(errors)];
        mean_acc = [mean_acc mean(accuracy)];
        mean_errs2 = mean_errs'
        mean_acc2 = mean_acc'
        final = [parameters; mean_errs2; mean_acc2];
        
        
        trainingError_mean(find(features==f),find(trees==t))=mean(trainingError);
        validationError_mean(find(features==f),find(trees==t))=mean(validationError);
        oobErr_mean(find(features==f),find(trees==t))=mean(oobErr);
        timeTaken_mean(find(features==f),find(trees==t))=mean(timeTaken);
        count = sprintf('%d, %d ',f,t)
    end
    ftur = f
end
toc;

% best accuracy
bestAccuracy = max(max(accuracy_mean));
        
%% PLOTTING THE GRID SEARCH
[features2, trees2] = ndgrid(features,trees);
% To plot oobErr, validationError and TrainingError
figure
surf(features2,trees2,oobErr_mean,'FaceColor','r','FaceAlpha',0.5);
hold on
surf(features2,trees2,validationError_mean,'FaceColor','b','FaceAlpha',0.5);
hold on
surf(features2,trees2,trainingError_mean,'FaceColor','g','FaceAlpha',0.5);
title('Random Forest - Optimisation for Hyperparameters')
xlabel('Number of Features')
ylabel('Number of Trees')
zlabel('Mean Error')
legend('OOB Erros', 'Validation Data', 'Training Data')
hold off

% Plotting the crossvalidation Accuracy Hyperparameter Grid
figure
surf(features2,trees2,accuracy_mean,'FaceColor', 'y', 'FaceAlpha', 0.5)
title('Random Forest (Classification) Accuracy based grid-search')
xlabel('Number of Features')
ylabel('Number of Trees')
zlabel('Percentage Score')
legend('Accuracy', 'F1-Score')
hold off

% Plotting the crossvalidation F1-score Hyperparameter Grid
figure
surf(features2,trees2,F1_crossval_Grid,'FaceColor','b','FaceAlpha',0.5)
title('Random Forest (Classification) F1-score based Grid Search')
xlabel('Number of Features')
ylabel('Number of Trees')
zlabel('Score')
legend('F1-score')
hold off

% Area Under the Precision Recall & FPTP Curve
auc_pr = max(max(AUC_crossval_Grid));
auc_z = max(max(AUC_FPTP_Grid));
Best_PR_Curve_Area = sprintf('%s %d ','Area under the Precision-Recall curve:',auc_pr)
Best_ROC_Curve_Area = sprintf('%s %d ','Area under the ROC curve:',auc_z)

% Time Plot

figure
surf(features2,trees2,timeTaken_mean,'FaceColor','c','FaceAlpha',0.5)
title('Random Forests - Optimisation for Time')
xlabel('Number of Features')
ylabel('Number of Trees')
zlabel('Time (mins)')
legend('Time (mins')
hold off


%% Cross Validation on Final Optimised Model --------------------------------------


final = array2table(final);
optimal_features = 5; % optimal hyperparameter combination from -
optimal_trees = 250; % - grid search graph plot 

% cross-validation on final optimised model to see if the performance
% differs from non-cross-validated final optimised model in the other file
% - (Final_Optimised_RF_Model.m)

aucpr_best = 0;
aucpr_mean = [];
auc_best = 0;
auc_mean = [];

mean_acc = [];
mean_err = [];
mean_f1score = [];

fold = 10;
for i=1:fold
    x2_fold = dataTrain_x(indx == i,:);
    x1_fold = dataTrain_x(indx ~= i,:);
    y_fold = dataTrain_y(indx ~= i,:);
    y2_fold = dataTrain_y(indx == i,:);
    idx_test = (indx == i);
            
            
    rng(1)
    RF_Optimised_Model = TreeBagger(optimal_trees, x1_fold, y_fold,...
        'NumPredictorsToSample', optimal_features, 'MinLeafSize', 5,...
        'Method', 'Classification', 'OOBPredictorImportance','on',...
        'PredictorSelection', 'curvature');

    [predLabelsFinal, scoresFinal] = predict(RF_Optimised_Model, x2_fold);

    % confusion matrix for optimised model
    y2_fold2 = num2cell( num2str(y2_fold) );
    cMFinalcv = confusionmat(y2_fold2, predLabelsFinal);
    % Performance
    [accur, f1score_final_cv, gmean_final] = performance(cMFinalcv);
    mean_acc(i) = 100*sum(diag(cMFinalcv))./sum(cMFinalcv(:));
    mean_err(i) = oobError(RF_Optimised_Model, 'Mode', 'Ensemble');
    mean_f1score(i) = f1score_final_cv;
    
    one_pos = find(strcmp('1',RF_Optimised_Model.ClassNames));
    
    [fp,tp,t,auc] = perfcurve(y2_fold2,scoresFinal(:,one_pos),'1');
    auc_mean(i) = auc;
    if auc > auc_best
        auc_best = auc;
        figure(30)
        plot(fp,tp)
        title('ROC Curve')
        xlabel('False positive rate')
        ylabel('True positive rate')
    end
    
    [x_pr, y_pr, t_pr, auc_pr] = perfcurve(y2_fold2, scoresFinal(:,one_pos),...
                '1', 'xCrit', 'reca', 'yCrit', 'prec');
    aucpr_mean(i) = auc_pr;
    if auc_pr > aucpr_best
        aucpr_best = auc_pr;
        figure(31)
        plot(x_pr, y_pr)
        title('Precision-Recall Curve')
        xlabel('Precision')
        ylabel('Recall')
    end
end
    
f1score_final = mean(mean_f1score)
acc_final = mean(mean_acc)
err_final = mean(mean_err)
best_auc = auc_best
best_aucpr = aucpr_best
auc_final = mean(auc_mean)
aucpr_final = mean(aucpr_mean)