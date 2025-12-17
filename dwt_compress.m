function approx = dwt_compress(ts, level, wname)
% DWT_COMPRESS  compute approximation coefficients at specified level
% Returns a 1D row vector (coeffs). If length small, returns original ts.

ts = ts(:)';
n = numel(ts);
if n < 8 || level <= 0
    approx = ts;
    return;
end

% Use wavedec to get approximation at level
try
    [C,L] = wavedec(ts, level, wname);
    % approximation coefficients are the first L(1) entries
    a = appcoef(C,L,wname,level);
    approx = a(:)';
catch
    % If wavelet toolbox not available, fallback to simple downsample smoothing
    approx = downsample(smoothdata(ts,'movmean',4), 4);
end
end
