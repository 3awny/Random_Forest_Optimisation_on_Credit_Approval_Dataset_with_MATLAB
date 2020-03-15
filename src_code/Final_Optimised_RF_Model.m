%% TEST UNSEEN DATA (FINAL PERFORMANCE)

%Splitting the Data to Test and Train
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

% FINAL MODEL ON TEST DATA
optimal_features = 5;
optimal_trees = 250;

rng(1)
RF_Optimised_Model_TEST = TreeBagger(optimal_trees, dataTrain_x, dataTrain_y,...
    'NumPredictorsToSample', optimal_features, 'MinLeafSize', 10,...
    'Method', 'Classification', 'OOBPredictorImportance','on',...
    'PredictorSelection', 'curvature');
[labelTestFinal, scoresFinalTest] = predict(RF_Optimised_Model_TEST, dataTest_x);

% Confusion Matrix
temp = cell2mat(labelTestFinal);
temp2 = str2num(temp);
c_mat_test = confusionmat(dataTest_y, temp2, 'Order',[1 0]);
figure;
c_chart_test = confusionchart(c_mat_test);
title('Random Forest Confusion Chart for Final Test Model')

% fscore & accuracy calculation
[accur_test, f1score_test, gmean_final_test] = performance(c_mat_test);
final_test_f1score = f1score_test
accuracy_test_final = accur_test

% AUC-PR
one_pos_test = find(strcmp('1',RF_Optimised_Model_TEST.ClassNames));
[x_pr_test, y_pr_test, t_pr_test, auc_pr_test] = perfcurve(dataTest_y, scoresFinalTest(:,one_pos_test),...
    '1', 'xCrit', 'reca', 'yCrit', 'prec');
% AUC
[fp_test,tp_test,t_test,auc_roc_test] = perfcurve(dataTest_y,scoresFinalTest(:,one_pos_test),'1');

% AUC PR and AUC ROC values
AUC_PR_FinalTest = auc_pr_test
auc_roc_FinalTest = auc_roc_test