function [trainFiles, testFiles] = stratified_split(fileIndex, train_ratio)
% STRATIFIED_SPLIT  split list preserving per-class ratio
labels = cellfun(@(x) x.label, fileIndex, 'UniformOutput', false);
uniqueLabels = unique(labels);

trainFiles = {};
testFiles = {};
rng(0); % for reproducibility
for i=1:numel(uniqueLabels)
    lab = uniqueLabels{i};
    idx = find(strcmp(labels, lab));
    idx = idx(randperm(numel(idx)));
    ntrain = max(1, round(train_ratio * numel(idx)));
    trainIdx = idx(1:ntrain);
    testIdx = idx(ntrain+1:end);
    trainFiles = [trainFiles, fileIndex(trainIdx)]; %#ok<AGROW>
    testFiles  = [testFiles,  fileIndex(testIdx)];  %#ok<AGROW>
end
end
