function evaluate_results(true_labels, pred_labels, classList)
% EVALUATE_RESULTS  print confusion matrix and metrics
% Inputs as cell arrays of strings

n = numel(true_labels);
if n~=numel(pred_labels)
    error('Length mismatch.');
end

% map labels to indices according to classList
C = numel(classList);
label2idx = containers.Map(classList, 1:C);
true_idx = zeros(n,1);
pred_idx = zeros(n,1);
true_idx = [];
pred_idx = [];

for i = 1:n
    t = string(true_labels{i});
    p = string(pred_labels{i});

    % Skip samples with invalid predictions
    if p == "unknown" || isempty(p)
        fprintf('[EVAL] Skip sample %d (pred=unknown)\n', i);
        continue;
    end

    if ~isKey(label2idx, t) || ~isKey(label2idx, p)
        fprintf('[EVAL] Skip sample %d (invalid label: %s -> %s)\n', i, t, p);
        continue;
    end

    true_idx(end+1) = label2idx(t);
    pred_idx(end+1) = label2idx(p);
end


cm = confusionmat(true_idx, pred_idx);
disp('Confusion matrix (rows=true, cols=pred):');
disp(array2table(cm, 'VariableNames', classList, 'RowNames', classList));

acc = sum(diag(cm))/sum(cm(:));
fprintf('Overall accuracy: %.3f\n', acc);

% per-class precision/recall/f1
prec = zeros(C,1); rec = zeros(C,1); f1 = zeros(C,1);
for k=1:C
    tp = cm(k,k);
    fp = sum(cm(:,k)) - tp;
    fn = sum(cm(k,:)) - tp;
    if (tp+fp)==0, prec(k)=0; else prec(k) = tp/(tp+fp); end
    if (tp+fn)==0, rec(k)=0; else rec(k) = tp/(tp+fn); end
    if (prec(k)+rec(k))==0, f1(k)=0; else f1(k) = 2*prec(k)*rec(k)/(prec(k)+rec(k)); end
    fprintf('Class %s: Prec=%.3f  Rec=%.3f  F1=%.3f\n', classList{k}, prec(k), rec(k), f1(k));
end


%% -------------------------------------------------------------
% 彩色混淆矩阵图（新增）
%% -------------------------------------------------------------

figure('Name','Confusion Matrix','NumberTitle','off');
imagesc(cm);       
colormap('jet');   % 彩色
colorbar;          % 颜色刻度条

% 设置刻度标签
xticks(1:C);
xticklabels(classList);
yticks(1:C);
yticklabels(classList);

xlabel('Predicted Class');
ylabel('True Class');
title('Confusion Matrix (DTW-KNN-SVM)');

% 数值写进格子里
for i = 1:C
    for j = 1:C
        text(j, i, num2str(cm(i,j)), ...
            'HorizontalAlignment', 'center', ...
            'Color', 'white', 'FontWeight', 'bold');
    end
end

axis square;


end
