function feats = extract_features_allpairs(X, config)
% EXTRACT_FEATURES_ALLPAIRS
% Return cell array with (Ntx*Nrx*config.pca_m) 1D compressed feature series

[T,F] = size(X);

% Deduce subcarrier count by factoring F
% We assume subcarriers count is 30 (Intel 5300 standard)
Nsc = 30;

% deduce number of antenna pairs
nPairs = F / Nsc;

if mod(F, Nsc) ~= 0
    error('Feature dimension F=%d is not divisible by Nsc=30. Check CSI loader.', F);
end

m = config.pca_m;               % number of PCA components per pair
perPair = Nsc;                  % subcarrier count per antenna pair
nFeatures = nPairs * m;         % total output feature channels

feats = cell(nFeatures, 1);
cnt = 0;

for p = 1:nPairs
    cols = ( (p-1)*perPair + 1 ) : (p*perPair);
    M = X(:, cols); % T x 30
    M = zscore_safe(M); % normalize

    % PCA
    [coeff, score, ~] = pca(M, 'NumComponents', m); % score = T x m

    for c = 1:m
        cnt = cnt + 1;
        ts = score(:, c); % 1D feature
        approx = dwt_compress(ts, config.dwt_level, config.dwt_wavelet);
        feats{cnt} = approx(:)';
    end
end



end

function Y = zscore_safe(X)
    mu = mean(X,1);
    sigma = std(X,0,1);
    sigma(sigma==0) = 1;
    Y = (X - mu) ./ sigma;
end

