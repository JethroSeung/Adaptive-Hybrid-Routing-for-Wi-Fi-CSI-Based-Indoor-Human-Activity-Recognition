function svm_model = train_svm_static_classifier(trainFiles)

fprintf('Training static SVM classifier...\n');

static_labels = {'stand','sit','null'};

Xtrain = [];
Ytrain = {};

for i = 1:numel(trainFiles)
    label = trainFiles{i}.label;
    if ~ismember(label, static_labels)
        continue;
    end

    try
        Xraw = load_sample(trainFiles{i}.path);
        feat = extract_static_features(Xraw);

        % only accept valid fixed-length vectors
        if isempty(feat) || any(isnan(feat))
            fprintf('[SKIP-invalid-static] %s\n', trainFiles{i}.path);
            continue;
        end

        Xtrain = [Xtrain; feat]; 
        Ytrain{end+1} = label;

    catch ME
        fprintf('[Error static sample] %s (%s)\n', ...
            trainFiles{i}.path, ME.message);
    end
end

if isempty(Xtrain)
    error('No valid static samples for SVM!');
end

Ytrain = categorical(Ytrain);

svm_model = fitcecoc(Xtrain, Ytrain);

fprintf('Static SVM training completed. Samples=%d, FeatureDim=%d\n', ...
    size(Xtrain,1), size(Xtrain,2));
end
