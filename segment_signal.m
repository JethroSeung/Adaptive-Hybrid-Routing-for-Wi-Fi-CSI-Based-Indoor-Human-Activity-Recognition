function segs = segment_signal(x, nSeg)
% SEGMENT_SIGNAL  Split vector x into nSeg equal segments
x = x(:)';
L = numel(x);
segLen = floor(L / nSeg);

segs = cell(nSeg,1);
for i = 1:nSeg
    s = (i-1)*segLen + 1;
    e = (i==nSeg) * L + (i<nSeg)*(i*segLen);
    segs{i} = x(s:e);
end
end
