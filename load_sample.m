function X = load_sample(fname)
% LOAD_SAMPLE robust loader for Intel 5300 CSI using read_bf_file()
% Automatically handles missing / corrupted packets and variable dimensional CSI.

csi_trace = read_bf_file(fname);
if isempty(csi_trace)
    error('read_bf_file returned empty for %s', fname);
end

num_packets = length(csi_trace);

% First valid entry to detect correct dimension
valid_found = false;
for i = 1:num_packets
    if ~isempty(csi_trace{i})
        tmp = get_scaled_csi(csi_trace{i});
        if ~isempty(tmp)
            [Ntx, Nrx, Nsc] = size(tmp);   % real dimension
            feat_dim = Ntx * Nrx * Nsc;
            valid_found = true;
            break;
        end
    end
end

if ~valid_found
    error('No valid CSI entry found in %s', fname);
end

% Pre-allocate based on detected dimension
X = zeros(num_packets, feat_dim);

valid_idx = 0;
for i = 1:num_packets
    entry = csi_trace{i};
    if isempty(entry)
        continue;
    end
    csi = get_scaled_csi(entry);
    if isempty(csi)
        continue;
    end
    try
        amp = abs(squeeze(csi));
        vec = amp(:)';
        if numel(vec) ~= feat_dim
            % Skip corrupted / dimension-changed packets
            continue;
        end
        valid_idx = valid_idx + 1;
        X(valid_idx,:) = vec;
    catch
        continue;
    end
end

% Trim to real valid packet count
X = X(1:valid_idx, :);

% If too short, skip or resample
target_len = 1000;
if size(X,1) < 50
    error('Too few valid CSI packets (<50) in %s', fname);
end

% Resample to fixed length
X = resample(X, target_len, size(X,1));

end
