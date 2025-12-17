function templates = build_templates(trainFiles, config)
% BUILD_TEMPLATES
% Construct DTW template library using training samples
% Robust to inconsistent feature counts (skips corrupted/incomplete samples)

fprintf('Building templates...\n');

%% Probe first valid sample to determine expected feature count
probeLoaded = false;

for i = 1:numel(trainFiles)
    try
        Xprobe = load_sample(trainFiles{i}.path);
        feats_probe = extract_features_allpairs(Xprobe, config);
        nFeatures = numel(feats_probe);
        probeLoaded = true;
        fprintf('Detected feature count per sample: %d\n', nFeatures);
        break;
    catch
        fprintf('[Probe Skip] %s\n', trainFiles{i}.path);
        continue;
    end
end

if ~probeLoaded
    error('Failed to detect feature dimension. No valid samples found.');
end

%% Pre-allocate template structure
templates = repmat(struct('label_list',{{}}, 'series', {{}}), nFeatures, 1);

%% Fill template library
for i = 1:numel(trainFiles)
    fname = trainFiles{i}.path;
    label = trainFiles{i}.label;

    try
        X = load_sample(fname);
        feats = extract_features_allpairs(X, config);

        % Skip if invalid feature length
        if numel(feats) ~= nFeatures
            fprintf('[SKIP] %s  (feature count %d != expected %d)\n', ...
                fname, numel(feats), nFeatures);
            continue;
        end

        % Add to template banks
        for f = 1:nFeatures
            templates(f).series{end+1} = feats{f};
            templates(f).label_list{end+1} = label;
        end

    catch ME
        fprintf('[ERROR SKIP] %s (%s)\n', fname, ME.message);
        continue;
    end
end

fprintf('Template building completed. Valid templates per feature: \n');
for f = 1:nFeatures
    fprintf('Feature %d : %d templates\n', f, numel(templates(f).label_list));
end

end
