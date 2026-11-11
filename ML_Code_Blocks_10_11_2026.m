
%% Code Blocks used in Machine Learning Analysis Section 

% ANALYSIS SETTINGS

% Define analysis parameters

nRepeats = 100;
trainRatio = 0.80;
nInnerFolds = 5;
pThreshold = 0.05; 


% Define SVM hyperparameter grid


CGrid = logspace(-3,3,7);
KernelScaleGrid = logspace(-2,2,9);


% Preallocate performance measures

accuracy = nan(nRepeats,1);
auc = nan(nRepeats,1);
sensitivity = nan(nRepeats,1);
specificity = nan(nRepeats,1);
selectedFeatureCount = nan(nRepeats,1);

lastModel = [];
lastTestLabels = categorical();
lastTestPredictions = categorical();
lastTestScores = [];
lastSelectedFeatures = [];
lastSelectedROINames = {};
selectedROIHistory = cell(nRepeats,1);


% REPEATED NESTED CROSS-VALIDATION

for r = 1:nRepeats

    fprintf('\nOuter repetition %d/%d\n',r,nRepeats);

    %% 3.1 Outer stratified train/test split

    outerCV = cvpartition(Y,'HoldOut',1-trainRatio);

    trainIdx = training(outerCV);
    testIdx = test(outerCV);

    Xtrain = X(trainIdx,:);
    Ytrain = Y(trainIdx);

    Xtest = X(testIdx,:);
    Ytest = Y(testIdx);



% Inner 5-fold cross-validation

    innerCV = cvpartition(Ytrain,'KFold',nInnerFolds);

    meanFoldLoss = inf( ...
        numel(CGrid), ...
        numel(KernelScaleGrid));

% Test every C and KernelScale combination

    for cIdx = 1:numel(CGrid)

        for sIdx = 1:numel(KernelScaleGrid)

            foldLoss = nan(nInnerFolds,1);


 % Split inner training and validation data

            for k = 1:nInnerFolds

                innerTrainIdx = training(innerCV,k);
                innerValidIdx = test(innerCV,k);

                XinnerTrain = Xtrain(innerTrainIdx,:);
                YinnerTrain = Ytrain(innerTrainIdx);

                XinnerValid = Xtrain(innerValidIdx,:);
                YinnerValid = Ytrain(innerValidIdx);

                % Hybrid feature selection

                % Feature selection is performed ONLY using
                % the current inner training fold.

                selected = hybridFeatureSelection( ...
                    XinnerTrain, ...
                    YinnerTrain, ...
                    pThreshold);


                % Train SVM

                svm = fitcsvm( ...
                    XinnerTrain(:,selected), ...
                    YinnerTrain, ...
                    'KernelFunction','rbf', ...
                    'BoxConstraint',CGrid(cIdx), ...
                    'KernelScale',KernelScaleGrid(sIdx), ...
                    'Standardize',true, ...
                    'ClassNames',classNames);


                % Evaluate on inner validation fold

                foldPred = predict( ...
                    svm, ...
                    XinnerValid(:,selected));

                foldLoss(k) = mean( ...
                    foldPred ~= YinnerValid);

            end

            % Calculate mean inner-CV loss

            meanFoldLoss(cIdx,sIdx) = mean(foldLoss);

        end
    end

    % Select optimal SVM hyperparameters

    [~,linearBest] = min(meanFoldLoss(:));

    [bestCIdx,bestScaleIdx] = ...
        ind2sub(size(meanFoldLoss),linearBest);

    bestC = CGrid(bestCIdx);

    bestKernelScale = ...
        KernelScaleGrid(bestScaleIdx);


 % Re-select features using the complete outer training set

    % The outer test set remains completely untouched.

    [selectedFeatures,selectedOrder] = ...
        hybridFeatureSelection( ...
            Xtrain, ...
            Ytrain, ...
            pThreshold);

    selectedFeatureCount(r) = ...
        numel(selectedFeatures);

    selectedROIHistory{r} = ...
        roiNames(selectedOrder);


  % Train final optimised SVM

    finalModel = fitcsvm( ...
        Xtrain(:,selectedFeatures), ...
        Ytrain, ...
        'KernelFunction','rbf', ...
        'BoxConstraint',bestC, ...
        'KernelScale',bestKernelScale, ...
        'Standardize',true, ...
        'ClassNames',classNames);


 % Evaluate final model on held-out test set

    [testPred,testScores] = predict( ...
        finalModel, ...
        Xtest(:,selectedFeatures));

    accuracy(r) = mean(testPred == Ytest);


  % Calculate ROC-AUC

    positiveIdx = find( ...
        finalModel.ClassNames == ...
        categorical({positiveClass}));

    [~,~,~,auc(r)] = perfcurve( ...
        Ytest, ...
        testScores(:,positiveIdx), ...
        positiveClass);

    % 3.6.2 Calculate sensitivity and specificity

    positiveMask = (Ytest == positiveClass);
    predictedPositive = (testPred == positiveClass);

    TP = sum(predictedPositive & positiveMask);
    FN = sum(~predictedPositive & positiveMask);
    TN = sum(~predictedPositive & ~positiveMask);
    FP = sum(predictedPositive & ~positiveMask);

    sensitivity(r) = TP/(TP+FN);
    specificity(r) = TN/(TN+FP);


   % Store final repetition

    if r == nRepeats

        lastModel = finalModel;
        lastTestLabels = Ytest;
        lastTestPredictions = testPred;
        lastTestScores = testScores;

        lastSelectedFeatures = selectedFeatures;
        lastSelectedROINames = roiNames(selectedOrder);
        lastSelectedROIOrder = roiNames(selectedOrder);

        lastBestC = bestC;
        lastBestKernelScale = bestKernelScale;

    end

end


% Confusion matrix


confusionchart( ...
    lastTestLabels, ...
    lastTestPredictions);

% ROC curve

[rocX,rocY,~,lastAUC] = perfcurve( ...
    lastTestLabels, ...
    lastTestScores(:, ...
    find(lastModel.ClassNames == ...
    categorical({positiveClass}))), ...
    positiveClass);


% SAVE MODEL AND RESULTS

% Select output file

[saveFile,savePath] = ...
    uiputfile('Best_SVM_Model.mat', ...
    'Save trained SVM model');

if ~isequal(saveFile,0)

 % Create performance table

    performance = table( ...
        (1:nRepeats)', ...
        accuracy, ...
        auc, ...
        sensitivity, ...
        specificity, ...
        selectedFeatureCount, ...
        'VariableNames', ...
        {'Repetition', ...
         'Accuracy', ...
         'AUC', ...
         'Sensitivity', ...
         'Specificity', ...
         'NSelectedROIs'});


% Save model, features, parameters and results

    save(fullfile(savePath,saveFile), ...
        'lastModel', ...
        'lastSelectedFeatures', ...
        'lastBestC', ...
        'lastBestKernelScale', ...
        'performance', ...
        'CGrid', ...
        'KernelScaleGrid', ...
        'pThreshold', ...
        'positiveClass', ...
        'roiNames', ...
        'lastSelectedROINames', ...
        'lastSelectedROIOrder', ...
        'selectedROIHistory');

end


% HYBRID FEATURE SELECTION FUNCTION

function [selectedGlobal,selectedOrder] = ...
    hybridFeatureSelection(Xdata,Ydata,pThreshold)


% Stage 1: Wilcoxon rank-sum screening

nFeatures = size(Xdata,2);

cats = categories(Ydata);

if numel(cats) ~= 2
    error('Hybrid ROI selection requires exactly two groups.');
end

group1 = (Ydata == cats{1});
group2 = (Ydata == cats{2});

pValues = ones(1,nFeatures);

for j = 1:nFeatures

    pValues(j) = ranksum( ...
        Xdata(group1,j), ...
        Xdata(group2,j));

end


% Retain ROIs with p < 0.05

candidateFeatures = ...
    find(pValues < pThreshold);

if isempty(candidateFeatures)

    warning( ...
        'No ROI passed p < %.3f in this training fold; retaining the ROI with the smallest p-value.', ...
        pThreshold);

    [~,candidateFeatures] = min(pValues);

end


% Stage 2: Sequential forward selection

Xcand = Xdata(:,candidateFeatures);

nPerClass = countcats(Ydata);

nFolds = min(3,min(nPerClass));


% Handle cases with one candidate ROI or insufficient folds

if numel(candidateFeatures) == 1 || nFolds < 2

    selectedOrder = candidateFeatures(:)';
    selectedGlobal = selectedOrder;

    return;

end


% Initialise sequential forward selection

fsCV = cvpartition(Ydata,'KFold',nFolds);

remaining = 1:numel(candidateFeatures);

currentOrder = [];

bestLoss = inf;

bestOrder = [];

tolerance = 1e-10;

% Evaluate candidate ROI additions

while ~isempty(remaining)

    candidateLoss = inf(size(remaining));

    for q = 1:numel(remaining)

        trialOrder = ...
            [currentOrder remaining(q)];

        foldLoss = nan(nFolds,1);

        for f = 1:nFolds

            tr = training(fsCV,f);
            va = test(fsCV,f);

            mdl = fitcsvm( ...
                Xcand(tr,trialOrder), ...
                Ydata(tr), ...
                'KernelFunction','rbf', ...
                'BoxConstraint',1, ...
                'KernelScale','auto', ...
                'Standardize',true, ...
                'ClassNames',cats);

            pred = predict( ...
                mdl, ...
                Xcand(va,trialOrder));

            foldLoss(f) = ...
                mean(pred ~= Ydata(va));

        end

        candidateLoss(q) = ...
            mean(foldLoss);

    end

% Select the ROI that gives the lowest CV error

    [stepLoss,bestCandidatePosition] = ...
        min(candidateLoss);

    if isempty(currentOrder) || ...
            stepLoss < bestLoss - tolerance

        chosen = ...
            remaining(bestCandidatePosition);

        currentOrder = ...
            [currentOrder chosen];

        bestOrder = currentOrder;

        bestLoss = stepLoss;

        remaining(bestCandidatePosition) = [];

    else

        break;

    end

end


%Return selected ROIs

selectedOrder = ...
    candidateFeatures(bestOrder);

selectedGlobal = selectedOrder;

end
