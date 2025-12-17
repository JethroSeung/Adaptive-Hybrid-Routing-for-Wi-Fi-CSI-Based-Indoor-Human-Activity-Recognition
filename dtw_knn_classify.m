function pred_label = dtw_knn_classify(feats, templates, K, config)
% Clean DTW-KNN classifier with:
%   - segment DTW (3 segments, weights 0.2, 0.6, 0.2)
%   - distance-weighted voting
%   - NO grid search, NO global override, NO parfor

if nargin < 4, config = struct(); end
if ~isfield(config,'dtw_window'), config.dtw_window = []; end

% 固定权重（从经验选取）
seg_w = [0.2 0.6 0.2];

nFeatures = numel(feats);
votes = [];          % Nx2 char cell
vote_weight = [];    % Nx1 numeric

for f = 1:nFeatures
    series_f = templates(f).series;
    labels_f = templates(f).label_list;
    nTemplates = numel(series_f);

    q = feats{f}(:)';     % query

    dists = zeros(nTemplates,1);

    % --- DTW matching (safe, sequential) ---
    for t = 1:nTemplates
        s = series_f{t}(:)';
        dtmp = seg_dtw_clean(q, s, seg_w, config.dtw_window);
        dists(t) = dtmp;
    end

    % --- Top-K neighbors ---
    ksel = min(K, numel(dists));
    [d_k, idx] = mink(dists, ksel);

    % --- Distance-weighted voting ---
    % weight = 1/(distance + epsilon)
    w = 1 ./ (d_k + 1e-6);

    for kk = 1:ksel
        votes{end+1} = labels_f{idx(kk)}; %#ok<AGROW>
        vote_weight(end+1) = w(kk); %#ok<AGROW>
    end
end

% ===== Weighted voting =====
pred_label = weighted_vote(votes, vote_weight);

end


% ================================================================
% Weighted voting
% ================================================================
function lab = weighted_vote(votes, weights)
    u = unique(votes);
    score = zeros(numel(u),1);
    for i = 1:numel(u)
        mask = strcmp(votes, u{i});
        score(i) = sum(weights(mask));
    end
    [~, idx] = max(score);
    lab = u{idx};
end


% ================================================================
% 3-segment DTW (clean, stable)
% ================================================================
function d = seg_dtw_clean(q, s, seg_w, win)
    Lq = numel(q);
    Ls = numel(s);
    if Lq < 3 || Ls < 3
        d = force_dtw_scalar(q, s, win);
        return;
    end

    % segmentation
    q1 = q(1:floor(Lq/3));
    q2 = q(floor(Lq/3)+1:floor(2*Lq/3));
    q3 = q(floor(2*Lq/3)+1:end);

    s1 = s(1:floor(Ls/3));
    s2 = s(floor(Ls/3)+1:floor(2*Ls/3));
    s3 = s(floor(2*Ls/3)+1:end);

    % DTW
    d1 = force_dtw_scalar(q1, s1, win);
    d2 = force_dtw_scalar(q2, s2, win);
    d3 = force_dtw_scalar(q3, s3, win);

    % weighted sum
    d = d1*seg_w(1) + d2*seg_w(2) + d3*seg_w(3);
end


% ================================================================
% Make sure dtw always returns a scalar
% ================================================================
function d = force_dtw_scalar(a, b, win)
    try
        if isempty(win)
            x = dtw(a,b);
        else
            x = dtw(a,b,win);
        end
    catch
        L = min(numel(a),numel(b));
        x = norm(a(1:L)-b(1:L));
    end

    if isscalar(x)
        d = double(x);
    else
        d = double(x(end));
    end
end
