function run_dtw_knn()
% RUN_DTW_KNN  Main entry to run DTW-KNN pipeline.
% Fix: 使用原始信号波动率(Raw Motion Energy)替代PCA方差，解决归一化导致动静无法分离的问题

% -------------------------
% 1. 环境初始化
% -------------------------
if isempty(gcp('nocreate'))
    try parpool; catch; end
end

% Add CSI tool
csi_extract_tool_path = 'D:\MatlabTools\CSI Extract Tool';
addpath(genpath(csi_extract_tool_path)); 

norm_str = @(s) string(strtrim(lower(char(s))));

%% Config
config.data_root = 'D:\ExperimentalData\WIFI_sensing_dataset\room1';
config.classes = {'bend','fall','null','run','sit','stand','walk','wave'};
config.train_ratio = 0.7;

config.pca_m = 2;
config.dwt_level = 2;
config.dwt_wavelet = 'db4';

config.k = 5;
config.verbose = true;
config.dtw_window = 80;     
config.use_gpu    = true;   

% === 标签定义 ===
raw_dynamic = {'bend','fall','run','walk','wave'};
raw_static  = {'stand','sit','null'};

dynamic_labels = norm_str(raw_dynamic);
static_labels  = norm_str(raw_static);

%% Gather file lists
fileIndex = list_dataset_files(config.data_root, config.classes);
if isempty(fileIndex)
    error('No files found.');
end

%% Train/test split
[trainFiles, testFiles] = stratified_split(fileIndex, config.train_ratio);

%% Build template library
% 建立 DTW 模板库 (只包含训练集中的样本)
templates = build_templates(trainFiles, config);
% 训练 SVM 静态分类器
svm_model = train_svm_static_classifier(trainFiles);

%% Classify test set
results = cell(numel(testFiles),1);
labels_true = cell(numel(testFiles),1);
route_log = cell(numel(testFiles),1); 

dynamic_vars = [];
static_vars  = [];



%% === 3. Adaptive Threshold Calibration (Robust Version) ===
fprintf('Calibrating thresholds adaptively based on Training Data...\n');

static_mei_values = [];

% 1. 收集所有静态样本的 MEI
for i = 1:numel(trainFiles)
    lab = norm_str(trainFiles{i}.label);
    if ismember(lab, static_labels)
        try
            X_calib = load_sample(trainFiles{i}.path);
            mei = mean(std(X_calib, 0, 1)); 
            static_mei_values(end+1) = mei;
        catch
            continue;
        end
    end
end

if isempty(static_mei_values)
    warning('No static samples found! Using fallback.');
    TH_LOW = 2.5; TH_HIGH = 6.0;
else
    % === 关键修改：剔除离群点 (Outlier Removal) ===
    % 很多时候静态数据里有几个大噪声，会把标准差拉得巨大，导致阈值虚高
    % 我们使用中位数绝对偏差 (MAD) 法或者是简单的分位数法来剔除
    
    % 方法：只保留 90% 的核心数据，剔除最大的 10% 异常值
    clean_statics = sort(static_mei_values);
    n_keep = floor(0.90 * numel(clean_statics)); 
    clean_statics = clean_statics(1:n_keep);
    
    mu_stat = mean(clean_statics);
    sigma_stat = std(clean_statics);
    
    % === 参数调整 ===
    % 之前 alpha=3.0 导致 Low Bound 太高 (~7.5)，吃掉了动态区间
    % 根据 Boxplot 观察，静态主体在 3.5 左右，动态从 4.0 开始
    % 所以我们需要更紧致的边界
    
    alpha = 1.5; % 90% 置信度 (从 3.0 降为 1.5)
    beta  = 5.0; % 动态边界 (从 6.0 降为 5.0)
    
    TH_LOW  = mu_stat + alpha * sigma_stat;
    
    % 保护机制：如果算出来的阈值太高（超过动态区的下限），强行压回去
    % 这里加一个经验上限 (Empirical Cap)，防止过拟合噪声
    if TH_LOW > 5.0
        fprintf('[Calibration Warning] Calculated TH_LOW %.2f is too high, clamped to 5.0\n', TH_LOW);
        TH_LOW = 5.0;
    end
    
    TH_HIGH = max(TH_LOW + 2.0, mu_stat + beta * sigma_stat);

    fprintf('Adaptive Calibration Result (Robust):\n');
    fprintf('  Static Mean (Clean): %.4f\n', mu_stat);
    fprintf('  Static Std  (Clean): %.4f\n', sigma_stat);
    fprintf('  Final TH_LOW : %.4f\n', TH_LOW);
    fprintf('  Final TH_HIGH: %.4f\n', TH_HIGH);
end

fprintf('\nStarting Classification Loop (Using Raw Motion Energy)...\n');

% ... (前面的代码不变) ...

fprintf('\nStarting Classification Loop (Grey Zone Strategy)...\n');

for i = 1:numel(testFiles)
    fname = testFiles{i}.path;
    raw_label = testFiles{i}.label;
    labels_true{i} = raw_label; 
    
    % === Robust loading, skip bad samples ===
    try
        X = load_sample(fname);
    catch ME
        fprintf('[SKIP BAD SAMPLE] %s  (%s)\n', fname, ME.message);
        results{i} = 'unknown';
        labels_true{i} = raw_label;
        route_log{i} = 'SKIPPED_BAD';
        continue;   % <<< 跳过这条，继续下一个文件
    end

    raw_std = std(X, 0, 1);
 
    v = mean(raw_std);       
    
    lab_norm = norm_str(raw_label);
    if ismember(lab_norm, dynamic_labels)
        dynamic_vars(end+1) = v; %#ok<AGROW>
    elseif ismember(lab_norm, static_labels)
        static_vars(end+1) = v;  %#ok<AGROW>
    end

    % =================================================
    % 3. Grey Zone 路由逻辑 (Final Fix)
    % =================================================
    
    if v >= TH_HIGH
        % High Energy -> Run, Bend, Strong Walk -> DTW
        feats = extract_features_allpairs(X, config); 
        pred = dtw_knn_classify(feats, templates, config.k, config);
        route_log{i} = 'DTW_High';
        
    elseif v <= TH_LOW
        % Low Energy -> Sit, Null -> SVM
        x_static = extract_static_features(X);   
        rawpred = predict(svm_model, x_static);
        pred = char(rawpred);             
        route_log{i} = 'SVM_Low';
        
    else
        % Grey Zone (2.5 ~ 6.0) -> Wave, Fall, Slow Walk, Stand
        
        feats = extract_features_allpairs(X, config);
        dtw_result = dtw_knn_classify(feats, templates, config.k, config);
        dtw_result_norm = norm_str(dtw_result);
        
        % === 逻辑修复 ===
        % 在模糊区，只要 DTW 判定是动态动作 (含 Walk)，都信任 DTW
        % 以前只信 wave/fall，导致 walk 被漏判
        dynamic_trust_list = ["wave", "fall", "walk", "run", "bend"]; 
        
        if ismember(dtw_result_norm, dynamic_trust_list)
            pred = dtw_result;
            route_log{i} = 'Grey_DTW'; 
        else
            % DTW 认为是静态，或者非常不像动态 -> 交给 SVM
            x_static = extract_static_features(X);
            rawpred = predict(svm_model, x_static);
            pred = char(rawpred);
            route_log{i} = 'Grey_SVM'; 
        end
    end

    results{i} = pred;
    
    if config.verbose
        fprintf('[%d/%d] True: %-6s | Pred: %-6s | Motion: %6.2f | Route: %s\n', ...
            i, numel(testFiles), raw_label, pred, v, route_log{i});
    end
end

% ... (循环结束后) ...

% 统计路由情况 (修复版)
fprintf('\n------------------------------------------------\n');
fprintf('Detailed Routing Stats:\n');
disp(tabulate(route_log)); % 显示详细的细分表格

% 计算大类汇总
count_dtw = sum(contains(route_log, 'DTW')); % 包含 DTW_High 和 Grey_DTW
count_svm = sum(contains(route_log, 'SVM')); % 包含 SVM_Low 和 Grey_SVM

fprintf('Summary:\n');
fprintf('Routed to DTW (Total): %d samples\n', count_dtw);
fprintf('Routed to SVM (Total): %d samples\n', count_svm);
fprintf('------------------------------------------------\n');

figure('Name', 'Grey Zone Analysis', 'Color', 'w', 'Position', [100 100 600 500]);

% ===== 绘箱线图 (Paper Publication Quality) =====
figure('Name', 'Grey Zone Analysis', 'Color', 'w', 'Position', [100 100 700 500]);

if ~isempty(dynamic_vars) && ~isempty(static_vars)
    vals = [dynamic_vars(:); static_vars(:)];
    groups = [repmat({'Dynamic Activities'}, numel(dynamic_vars), 1);
              repmat({'Static Activities'},  numel(static_vars),  1)];

    % 1. 绘制灰色背景 (Grey Zone)
    % 放在最底层
    grid on; hold on;
    x_lims = [0.5, 2.5]; % boxplot 只有两组，x轴通常是 1 和 2
    fill([x_lims(1) x_lims(2) x_lims(2) x_lims(1)], ...
         [TH_LOW TH_LOW TH_HIGH TH_HIGH], ...
         [0.92 0.92 0.92], 'EdgeColor', 'none', 'FaceAlpha', 0.8);

    % 2. 绘制箱线图 (使用更细的线和颜色)
    % 这是一个小技巧：先画空的 boxplot 占位
    h = boxplot(vals, groups, 'Widths', 0.4, 'Symbol', 'k+', 'Colors', 'k');
    set(h, 'LineWidth', 1.2); % 加粗线条，看起来更清晰
    
    ylabel('Motion Energy Index (Mean Std)', 'FontName', 'Times New Roman', 'FontSize', 12);
    title('Adaptive Routing Strategy via Grey Zone', 'FontName', 'Times New Roman', 'FontSize', 14, 'FontWeight', 'bold');
    
    % 3. 绘制阈值线
    yline(TH_LOW,  'b--', 'Static Bound', 'LineWidth', 2, 'FontName', 'Times New Roman', 'FontSize', 10, 'LabelHorizontalAlignment', 'left');
    yline(TH_HIGH, 'r--', 'Dynamic Bound', 'LineWidth', 2, 'FontName', 'Times New Roman', 'FontSize', 10, 'LabelHorizontalAlignment', 'left');
    
    % 4. 添加核心说明文字 (论文的故事核心)
    text(1.5, (TH_LOW+TH_HIGH)/2, {'Grey Zone', '(Hybrid Verification)'}, ...
        'HorizontalAlignment', 'center', ...
        'FontName', 'Times New Roman', 'FontSize', 11, 'FontAngle', 'italic', 'Color', [0.3 0.3 0.3]);
    
    % 5. 调整坐标轴字体
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);
    xlim([0.5, 2.5]); % 锁定 x 轴范围
    ylim([0, 14]);    % 根据你的数据范围锁定 y 轴，看起来更整洁
    
    hold off;
end
%% Evaluate
evaluate_results(labels_true, results, config.classes);


%% ======== Plot Per-Class Accuracy (Publication Quality) ========
fprintf('Generating per-class accuracy bar chart...\n');

% Convert true & predicted labels to categorical arrays
true_labels = categorical(labels_true);
pred_labels = categorical(results);

class_list = categorical(config.classes);
num_classes = numel(class_list);

per_class_acc = zeros(num_classes,1);

% Compute accuracy per class manually
for ci = 1:num_classes
    cls = class_list(ci);
    idx = (true_labels == cls);
    if sum(idx)==0
        per_class_acc(ci) = NaN;
    else
        per_class_acc(ci) = sum(pred_labels(idx)==cls) / sum(idx);
    end
end

% ----- Plot -----
figure('Name','Per-Class Accuracy','Color','w','Position',[200 150 700 450]);
b = bar(per_class_acc*100, 'FaceColor',[0.25 0.45 0.75]);  % 高级蓝色

% Font + Style (论文级)
set(gca,'FontName','Times New Roman','FontSize',12, ...
        'XTick',1:num_classes, 'XTickLabel', cellstr(class_list));

ylabel('Accuracy (%)','FontName','Times New Roman','FontSize',14);
title('Per-Class Recognition Accuracy','FontName','Times New Roman', ...
      'FontSize',16,'FontWeight','bold');

ylim([0 110]);
grid on;

% Add accuracy text on each bar
for i = 1:num_classes
    val = per_class_acc(i)*100;
    text(i, val + 3, sprintf('%.1f', val), ...
         'HorizontalAlignment','center', ...
         'FontName','Times New Roman','FontSize',11);
end


end