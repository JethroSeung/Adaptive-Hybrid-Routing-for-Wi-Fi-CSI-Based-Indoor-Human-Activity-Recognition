function feat = extract_static_features(X)
% Robust static feature extractor
% Ensures fixed-length output by padding or truncation
% Used for SVM static classifier

TARGET_DIM = 600;   % <<< 统一维度，可按需修改

% --- Basic statistics ---
mu  = mean(X, 1);       % 1×F
sd  = std(X, 0, 1);     % 1×F
energy = sum(X.^2, 1);  % 1×F

% --- Low-frequency PSD ---
F = size(X,2);
psd_low = zeros(1,F);
for i = 1:F
    xd = detrend(X(:,i));
    xd = xd - mean(xd);
    Y = abs(fft(xd)).^2;
    psd_low(i) = sum(Y(1:5));
end

% --- Concatenate ---
feat = [mu, sd, energy, psd_low];   % 1×(4F)

% --- Dataset-level normalization (sample-wise is fine here) ---
feat = feat(:)';
feat = zscore(feat);

% === Robust dimension alignment ===
curDim = numel(feat);

if curDim > TARGET_DIM
    % truncate
    feat = feat(1:TARGET_DIM);

elseif curDim < TARGET_DIM
    % zero-pad
    feat = [feat, zeros(1, TARGET_DIM - curDim)];
end

end
