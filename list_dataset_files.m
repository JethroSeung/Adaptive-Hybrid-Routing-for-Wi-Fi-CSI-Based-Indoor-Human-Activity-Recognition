function fileIndex = list_dataset_files(data_root, classes)
% LIST_DATASET_FILES 
% Only extract files from volunteer "1102_xy"
% and sort files by the numeric index at the end of filename.

disp('>>> USING MY CUSTOM list_dataset_files !!!');

fileIndex = {};
idx = 0;

targetID = '1102_xy';   % *** target volunteer ***

for c = 1:numel(classes)
    className = classes{c};
    folder = fullfile(data_root, className);

    if ~isfolder(folder)
        warning('Folder not found: %s (skipping)', folder);
        continue;
    end

    % list all .dat files
    f = dir(fullfile(folder, '*.dat'));
    if isempty(f)
        warning('No .dat files in %s', folder);
        continue;
    end

    % ------------------------------------------------------
    % Keep only files from 1102_xy
    % ------------------------------------------------------
    keep = arrayfun(@(x) contains(x.name, targetID), f);
    f = f(keep);

    if isempty(f)
        warning('No files from %s in %s', targetID, folder);
        continue;
    end

    % ------------------------------------------------------
    % Extract numeric index at end of filename
    % e.g. 1102_xy_bend_55.dat --> 55
    % ------------------------------------------------------
    nums = zeros(numel(f), 1);
    for i = 1:numel(f)
        tokens = regexp(f(i).name, '(\d+)\.dat$', 'tokens');
        if ~isempty(tokens)
            nums(i) = str2double(tokens{1}{1});
        else
            nums(i) = inf;    % very unlikely now
        end
    end

    % numeric sort
    [~, order] = sort(nums);
    f = f(order);

    % ------------------------------------------------------
    % Only take first 100
    % ------------------------------------------------------
    N = min(100, numel(f));

    for i = 1:N
        idx = idx + 1;
        fileIndex{idx}.path = fullfile(folder, f(i).name);
        fileIndex{idx}.label = className;
    end
end

end
