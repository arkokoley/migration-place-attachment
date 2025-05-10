function analyzePlaceAttachmentResponses()
%=========================================================================%
%  MIDAS Place Attachment Analysis Script (Simplified)
%
%  This script analyzes how different place attachment levels affect:
%  1. Total migrations over time
%  2. In-migrations to shocked vs. non-shocked areas
%  3. Out-migrations from shocked vs. non-shocked areas  
%  4. Average wealth over time
%  5. Population in shocked vs. non-shocked areas over time
%  6. Average income over time
%
%  It compares:
%  - No place attachment (control)
%  - Low place attachment
%  - Medium place attachment
%  - High place attachment
%=========================================================================%

%% 1. Folder layout and parameters ---------------------------------------
experimentDir = './Outputs/shock_placeattachment_experiment/';  % Input directory
outputDir     = './Outputs/placeattachment_analysis/';          % Output directory
if ~exist(outputDir,'dir');  mkdir(outputDir);  end

% Loading experiment parameters from the input summary file
fprintf('Loading experiment parameters...\n');
summaryFiles = dir(fullfile(experimentDir,'PA_Shock_PlaceAttachment_*_input_summary.mat'));
if isempty(summaryFiles)
    error('No input summary file found! Run experiments first.');
end

% Load most recent summary file
[~, idx] = max([summaryFiles.datenum]);
paramFile = fullfile(summaryFiles(idx).folder, summaryFiles(idx).name);
params = load(paramFile);

% Experiment parameters
conditionLabels = params.conditionLabels;
shockStart    = 31;                             % First step of the shock
shockDur      = 30;                             % Length of shock
shockEnd      = shockStart + shockDur - 1;      % Last step of shock
Tmax          = 78;                             % Total timesteps in simulation

fprintf('Loading experiment files from %s\n', experimentDir);
outputFiles = dir(fullfile(experimentDir,'PA_Shock_PlaceAttachment_*.mat'));
outputFiles = outputFiles(~contains({outputFiles.name}, 'input_summary'));
fprintf('Found %d files...\n', numel(outputFiles));

%% 2. Pre-allocate metric storage arrays ---------------------------------
% For each place attachment level, we'll collect all runs
% 1st dimension: place attachment condition index
% 2nd dimension: run number (up to max runs per condition)
% 3rd dimension: time step

% First count how many runs we have for each condition
maxRuns = 10;  % Assume at most 10 runs per condition
numRuns = zeros(length(conditionLabels), 1);

% Initialize data arrays for each metric
totalMigrations = nan(length(conditionLabels), maxRuns, Tmax);
inMigrationsShocked = nan(length(conditionLabels), maxRuns, Tmax);
inMigrationsNonShocked = nan(length(conditionLabels), maxRuns, Tmax);
outMigrationsShocked = nan(length(conditionLabels), maxRuns, Tmax);
outMigrationsNonShocked = nan(length(conditionLabels), maxRuns, Tmax);
avgWealth = nan(length(conditionLabels), maxRuns, Tmax);

% Arrays for income metrics (only average)
avgIncome = nan(length(conditionLabels), maxRuns, Tmax);

% Arrays for population tracking in shocked vs non-shocked areas
popShocked = nan(length(conditionLabels), maxRuns, Tmax);
popNonShocked = nan(length(conditionLabels), maxRuns, Tmax);

%% 3. Process each output file ------------------------------------------
for k = 1:numel(outputFiles)
    fName = outputFiles(k).name;
    fPath = fullfile(outputFiles(k).folder, fName);
    
    % Try to load the file
    try
        S = load(fPath);
    catch
        warning('   – %s is corrupt... skipping', fName);
        continue;
    end
    
    if ~isfield(S, 'output')
        warning('   – no "output" field in %s', fName);
        continue;
    end
    
    out = S.output;
    
    % Get place attachment condition for this run
    [paCondition, runNumber] = getPlaceAttachmentCondition(S, fName, conditionLabels);
    
    if isempty(paCondition)
        warning('   – %s has unrecognized condition... skipping', fName);
        continue;
    end
    
    % Increment run counter for this condition
    if runNumber > maxRuns
        warning('More than %d runs found for condition %s, ignoring extras', maxRuns, conditionLabels{paCondition});
        continue;
    end
    
    fprintf('   • Processing %-55s  condition=%s  (run %d)\n', fName, conditionLabels{paCondition}, runNumber);
    
    %% 3.1 Extract total migrations ------------------------------------
    if isfield(out, 'migrations')
        mig = out.migrations(:);
        totalMigrations(paCondition, runNumber, 1:length(mig)) = mig;
    end
    
    %% 3.2 Extract average wealth --------------------------------------
    if isfield(out, 'averageWealth')
        wealth = out.averageWealth(:);
        avgWealth(paCondition, runNumber, 1:length(wealth)) = wealth;
    end
    
    %% 3.3 Extract average income -----------------------------
    if isfield(out, 'avgIncome')
        incomeAvg = out.avgIncome(:);
        avgIncome(paCondition, runNumber, 1:length(incomeAvg)) = incomeAvg;
    end
    
    %% 3.4 Extract population in shocked vs non-shocked areas -------
    if isfield(out, 'agentsPerLocation') && isfield(out, 'mapVariables')
        % Get locations and identify which are shocked
        locations = out.mapVariables.locations;
        shocked = locations.locationY < 350;  % Locations affected by shock
        
        % Get population per location over time
        agentsPerLocation = out.agentsPerLocation;
        
        % Sum up population in shocked and non-shocked areas for each time step
        popShockedData = zeros(1, size(agentsPerLocation, 2));
        popNonShockedData = zeros(1, size(agentsPerLocation, 2));
        
        for t = 1:size(agentsPerLocation, 2)
            popShockedData(t) = sum(agentsPerLocation(shocked, t));
            popNonShockedData(t) = sum(agentsPerLocation(~shocked, t));
        end
        
        % Store the population data
        popShocked(paCondition, runNumber, 1:length(popShockedData)) = popShockedData;
        popNonShocked(paCondition, runNumber, 1:length(popNonShockedData)) = popNonShockedData;
    end
    
    %% 3.5 Extract in/out migrations by shock status -------------------
    if isfield(out, 'migrationMatrix') && isfield(out, 'mapVariables')
        % Identify shocked locations
        locations = out.mapVariables.locations;
        shocked = locations.locationY < 350;  % Locations affected by shock
        
        % Get migration matrix over time
        migMatrix = out.migrationMatrix;
        
        % Pre-allocate arrays for this run
        inShocked = zeros(1, Tmax);
        inNonShocked = zeros(1, Tmax);
        outShocked = zeros(1, Tmax);
        outNonShocked = zeros(1, Tmax);
        
        % Process each time step
        for t = 1:size(migMatrix, 3)
            % Migration matrix has dimensions: from_location × to_location × time
            mm = squeeze(migMatrix(:,:,t));
            
            % In-migrations to shocked areas: sum of column for shocked locations
            inShocked(t) = sum(sum(mm(:, shocked)));
            
            % In-migrations to non-shocked areas: sum of column for non-shocked locations
            inNonShocked(t) = sum(sum(mm(:, ~shocked)));
            
            % Out-migrations from shocked areas: sum of row for shocked locations
            outShocked(t) = sum(sum(mm(shocked, :)));
            
            % Out-migrations from non-shocked areas: sum of row for non-shocked locations
            outNonShocked(t) = sum(sum(mm(~shocked, :)));
        end
        
        % Store in our arrays
        inMigrationsShocked(paCondition, runNumber, 1:length(inShocked)) = inShocked;
        inMigrationsNonShocked(paCondition, runNumber, 1:length(inNonShocked)) = inNonShocked;
        outMigrationsShocked(paCondition, runNumber, 1:length(outShocked)) = outShocked;
        outMigrationsNonShocked(paCondition, runNumber, 1:length(outNonShocked)) = outNonShocked;
    end
end

%% 4. Calculate statistics for each place attachment level -----------------------
% For each metric and each condition, calculate:
% - Mean across runs
% - Upper bound (mean + std)
% - Lower bound (mean - std)

% Initialize arrays for means and bounds
meanTotalMigrations = zeros(length(conditionLabels), Tmax);
upperTotalMigrations = zeros(length(conditionLabels), Tmax);
lowerTotalMigrations = zeros(length(conditionLabels), Tmax);

meanInMigrationsShocked = zeros(length(conditionLabels), Tmax);
upperInMigrationsShocked = zeros(length(conditionLabels), Tmax);
lowerInMigrationsShocked = zeros(length(conditionLabels), Tmax);

meanInMigrationsNonShocked = zeros(length(conditionLabels), Tmax);
upperInMigrationsNonShocked = zeros(length(conditionLabels), Tmax);
lowerInMigrationsNonShocked = zeros(length(conditionLabels), Tmax);

meanOutMigrationsShocked = zeros(length(conditionLabels), Tmax);
upperOutMigrationsShocked = zeros(length(conditionLabels), Tmax);
lowerOutMigrationsShocked = zeros(length(conditionLabels), Tmax);

meanOutMigrationsNonShocked = zeros(length(conditionLabels), Tmax);
upperOutMigrationsNonShocked = zeros(length(conditionLabels), Tmax);
lowerOutMigrationsNonShocked = zeros(length(conditionLabels), Tmax);

meanAvgWealth = zeros(length(conditionLabels), Tmax);
upperAvgWealth = zeros(length(conditionLabels), Tmax);
lowerAvgWealth = zeros(length(conditionLabels), Tmax);

meanAvgIncome = zeros(length(conditionLabels), Tmax);
upperAvgIncome = zeros(length(conditionLabels), Tmax);
lowerAvgIncome = zeros(length(conditionLabels), Tmax);

meanPopShocked = zeros(length(conditionLabels), Tmax);
upperPopShocked = zeros(length(conditionLabels), Tmax);
lowerPopShocked = zeros(length(conditionLabels), Tmax);

meanPopNonShocked = zeros(length(conditionLabels), Tmax);
upperPopNonShocked = zeros(length(conditionLabels), Tmax);
lowerPopNonShocked = zeros(length(conditionLabels), Tmax);

% Calculate statistics for each condition
for ri = 1:length(conditionLabels)
    validRunCount = sum(~isnan(totalMigrations(ri, :, 1)));
    fprintf('Condition %s: %d valid runs\n', conditionLabels{ri}, validRunCount);
    
    if validRunCount == 0
        continue;
    end
    
    % Total Migrations
    validRuns = squeeze(~isnan(totalMigrations(ri, 1:maxRuns, :)));
    for t = 1:Tmax
        validData = squeeze(totalMigrations(ri, validRuns(:, t), t));
        if ~isempty(validData)
            meanTotalMigrations(ri, t) = mean(validData);
            stdDev = std(validData);
            upperTotalMigrations(ri, t) = meanTotalMigrations(ri, t) + stdDev;
            lowerTotalMigrations(ri, t) = max(0, meanTotalMigrations(ri, t) - stdDev);
        end
    end
    
    % In-Migrations to Shocked Areas
    validRuns = squeeze(~isnan(inMigrationsShocked(ri, 1:maxRuns, :)));
    for t = 1:Tmax
        validData = squeeze(inMigrationsShocked(ri, validRuns(:, t), t));
        if ~isempty(validData)
            meanInMigrationsShocked(ri, t) = mean(validData);
            stdDev = std(validData);
            upperInMigrationsShocked(ri, t) = meanInMigrationsShocked(ri, t) + stdDev;
            lowerInMigrationsShocked(ri, t) = max(0, meanInMigrationsShocked(ri, t) - stdDev);
        end
    end
    
    % In-Migrations to Non-Shocked Areas
    validRuns = squeeze(~isnan(inMigrationsNonShocked(ri, 1:maxRuns, :)));
    for t = 1:Tmax
        validData = squeeze(inMigrationsNonShocked(ri, validRuns(:, t), t));
        if ~isempty(validData)
            meanInMigrationsNonShocked(ri, t) = mean(validData);
            stdDev = std(validData);
            upperInMigrationsNonShocked(ri, t) = meanInMigrationsNonShocked(ri, t) + stdDev;
            lowerInMigrationsNonShocked(ri, t) = max(0, meanInMigrationsNonShocked(ri, t) - stdDev);
        end
    end
    
    % Out-Migrations from Shocked Areas
    validRuns = squeeze(~isnan(outMigrationsShocked(ri, 1:maxRuns, :)));
    for t = 1:Tmax
        validData = squeeze(outMigrationsShocked(ri, validRuns(:, t), t));
        if ~isempty(validData)
            meanOutMigrationsShocked(ri, t) = mean(validData);
            stdDev = std(validData);
            upperOutMigrationsShocked(ri, t) = meanOutMigrationsShocked(ri, t) + stdDev;
            lowerOutMigrationsShocked(ri, t) = max(0, meanOutMigrationsShocked(ri, t) - stdDev);
        end
    end
    
    % Out-Migrations from Non-Shocked Areas
    validRuns = squeeze(~isnan(outMigrationsNonShocked(ri, 1:maxRuns, :)));
    for t = 1:Tmax
        validData = squeeze(outMigrationsNonShocked(ri, validRuns(:, t), t));
        if ~isempty(validData)
            meanOutMigrationsNonShocked(ri, t) = mean(validData);
            stdDev = std(validData);
            upperOutMigrationsNonShocked(ri, t) = meanOutMigrationsNonShocked(ri, t) + stdDev;
            lowerOutMigrationsNonShocked(ri, t) = max(0, meanOutMigrationsNonShocked(ri, t) - stdDev);
        end
    end
    
    % Average Wealth
    validRuns = squeeze(~isnan(avgWealth(ri, 1:maxRuns, :)));
    for t = 1:Tmax
        validData = squeeze(avgWealth(ri, validRuns(:, t), t));
        if ~isempty(validData)
            meanAvgWealth(ri, t) = mean(validData);
            stdDev = std(validData);
            upperAvgWealth(ri, t) = meanAvgWealth(ri, t) + stdDev;
            lowerAvgWealth(ri, t) = meanAvgWealth(ri, t) - stdDev;
        end
    end
    
    % Average Income
    validRuns = squeeze(~isnan(avgIncome(ri, 1:maxRuns, :)));
    for t = 1:Tmax
        validData = squeeze(avgIncome(ri, validRuns(:, t), t));
        if ~isempty(validData)
            meanAvgIncome(ri, t) = mean(validData);
            stdDev = std(validData);
            upperAvgIncome(ri, t) = meanAvgIncome(ri, t) + stdDev;
            lowerAvgIncome(ri, t) = meanAvgIncome(ri, t) - stdDev;
        end
    end
    
    % Population in Shocked Areas
    validRuns = squeeze(~isnan(popShocked(ri, 1:maxRuns, :)));
    for t = 1:Tmax
        validData = squeeze(popShocked(ri, validRuns(:, t), t));
        if ~isempty(validData)
            meanPopShocked(ri, t) = mean(validData);
            stdDev = std(validData);
            upperPopShocked(ri, t) = meanPopShocked(ri, t) + stdDev;
            lowerPopShocked(ri, t) = meanPopShocked(ri, t) - stdDev;
        end
    end
    
    % Population in Non-Shocked Areas
    validRuns = squeeze(~isnan(popNonShocked(ri, 1:maxRuns, :)));
    for t = 1:Tmax
        validData = squeeze(popNonShocked(ri, validRuns(:, t), t));
        if ~isempty(validData)
            meanPopNonShocked(ri, t) = mean(validData);
            stdDev = std(validData);
            upperPopNonShocked(ri, t) = meanPopNonShocked(ri, t) + stdDev;
            lowerPopNonShocked(ri, t) = meanPopNonShocked(ri, t) - stdDev;
        end
    end
end

%% 5. Save analyzed data ------------------------------------------------
save(fullfile(outputDir, 'placeattachment_response_analysis.mat'), ...
    'conditionLabels', 'shockStart', 'shockDur', ...
    'meanTotalMigrations', 'upperTotalMigrations', 'lowerTotalMigrations', ...
    'meanInMigrationsShocked', 'upperInMigrationsShocked', 'lowerInMigrationsShocked', ...
    'meanInMigrationsNonShocked', 'upperInMigrationsNonShocked', 'lowerInMigrationsNonShocked', ...
    'meanOutMigrationsShocked', 'upperOutMigrationsShocked', 'lowerOutMigrationsShocked', ...
    'meanOutMigrationsNonShocked', 'upperOutMigrationsNonShocked', 'lowerOutMigrationsNonShocked', ...
    'meanAvgWealth', 'upperAvgWealth', 'lowerAvgWealth', ...
    'meanAvgIncome', 'upperAvgIncome', 'lowerAvgIncome', ...
    'meanPopShocked', 'upperPopShocked', 'lowerPopShocked', ...
    'meanPopNonShocked', 'upperPopNonShocked', 'lowerPopNonShocked');

%% 6. Create visualizations --------------------------------------------
% Time steps for x-axis
timeSteps = 1:Tmax;

% Create color scheme for conditions
colors = [
    0.0, 0.0, 0.0;  % black for none
    0.0, 0.5, 1.0;  % blue for low
    0.0, 0.8, 0.3;  % green for medium 
    1.0, 0.2, 0.0   % red for high
];

% 1. Plot total migrations
plotMetricWithBounds(timeSteps, meanTotalMigrations, upperTotalMigrations, lowerTotalMigrations, ...
    conditionLabels, colors, [shockStart, shockEnd], ...
    'Total Migrations by Place Attachment Level', 'Number of Migrations', ...
    fullfile(outputDir, 'total_migrations.png'));

% 2. Plot out-migrations from shocked areas
plotMetricWithBounds(timeSteps, meanOutMigrationsShocked, upperOutMigrationsShocked, lowerOutMigrationsShocked, ...
    conditionLabels, colors, [shockStart, shockEnd], ...
    'Out-Migrations from Shocked Areas by Place Attachment Level', 'Number of Migrations', ...
    fullfile(outputDir, 'out_migrations_shocked.png'));

% 3. Plot in-migrations to non-shocked areas
plotMetricWithBounds(timeSteps, meanInMigrationsNonShocked, upperInMigrationsNonShocked, lowerInMigrationsNonShocked, ...
    conditionLabels, colors, [shockStart, shockEnd], ...
    'In-Migrations to Non-Shocked Areas by Place Attachment Level', 'Number of Migrations', ...
    fullfile(outputDir, 'in_migrations_non_shocked.png'));

% 4. Plot average wealth
plotMetricWithBounds(timeSteps, meanAvgWealth, upperAvgWealth, lowerAvgWealth, ...
    conditionLabels, colors, [shockStart, shockEnd], ...
    'Average Wealth by Place Attachment Level', 'Average Wealth', ...
    fullfile(outputDir, 'average_wealth.png'));

% 5. Plot average income
plotMetricWithBounds(timeSteps, meanAvgIncome, upperAvgIncome, lowerAvgIncome, ...
    conditionLabels, colors, [shockStart, shockEnd], ...
    'Average Income by Place Attachment Level', 'Average Income', ...
    fullfile(outputDir, 'average_income.png'));

% 6. Plot population in shocked areas
plotMetricWithBounds(timeSteps, meanPopShocked, upperPopShocked, lowerPopShocked, ...
    conditionLabels, colors, [shockStart, shockEnd], ...
    'Population in Shocked Areas by Place Attachment Level', 'Number of Agents', ...
    fullfile(outputDir, 'population_shocked.png'));

% 7. Plot population in non-shocked areas
plotMetricWithBounds(timeSteps, meanPopNonShocked, upperPopNonShocked, lowerPopNonShocked, ...
    conditionLabels, colors, [shockStart, shockEnd], ...
    'Population in Non-Shocked Areas by Place Attachment Level', 'Number of Agents', ...
    fullfile(outputDir, 'population_nonshocked.png'));

% 8. Plot comparative population distribution before, during, and after shock
plotPopulationDistribution(meanPopShocked, meanPopNonShocked, ...
    conditionLabels, colors, shockStart, shockEnd, ...
    fullfile(outputDir, 'population_distribution.png'));

fprintf('\n✅ Analysis complete! Results saved to %s\n', outputDir);
end

%% ===================== Helper Functions ===============================

function [condition, runNumber] = getPlaceAttachmentCondition(S, fName, conditionLabels)
    % Extract place attachment condition from the file
    
    % Try to parse from filename
    for i = 1:length(conditionLabels)
        pattern = ['pa_' conditionLabels{i} '_rep(\d+)'];
        tok = regexp(fName, pattern, 'tokens', 'once');
        if ~isempty(tok)
            condition = i;
            runNumber = str2double(tok{1});
            return;
        end
    end
    
    % If we get here, try from the input parameters
    if isfield(S, 'input')
        pn = S.input.parameterNames;
        pv = S.input.parameterValues;
        
        % Check if place attachment flag is present
        flagIdx = find(strcmp(pn, 'modelParameters.placeAttachmentFlag'));
        if ~isempty(flagIdx)
            paFlag = pv{flagIdx};
            
            if paFlag == 0
                condition = 1;  % 'none' condition
                
                % Extract run number
                tok = regexp(fName, 'rep(\d+)', 'tokens', 'once');
                if ~isempty(tok)
                    runNumber = str2double(tok{1});
                else
                    runNumber = 1;  % Default if we can't find it
                end
                return;
            else
                % If place attachment is enabled, look for the mean value
                meanIdx = find(strcmp(pn, 'agentParameters.placeAttachmentMean'));
                if ~isempty(meanIdx)
                    paMean = pv{meanIdx};
                    
                    % Determine which condition based on mean value
                    if paMean < 0.35
                        condition = 2;  % 'low' condition
                    elseif paMean < 0.65
                        condition = 3;  % 'med' condition
                    else
                        condition = 4;  % 'high' condition
                    end
                    
                    % Extract run number
                    tok = regexp(fName, 'rep(\d+)', 'tokens', 'once');
                    if ~isempty(tok)
                        runNumber = str2double(tok{1});
                    else
                        runNumber = 1;  % Default if we can't find it
                    end
                    return;
                end
            end
        end
    end
    
    % Try to extract from run number in filename
    tok = regexp(fName, 'run(\d+)', 'tokens', 'once');
    if ~isempty(tok)
        runNumber = str2double(tok{1});
        
        % Try to determine condition from filename or modelParameters.shortName
        if contains(lower(fName), 'pa_none')
            condition = 1;  % none
            return;
        elseif contains(lower(fName), 'pa_low')
            condition = 2;  % low
            return;
        elseif contains(lower(fName), 'pa_med')
            condition = 3;  % medium
            return;
        elseif contains(lower(fName), 'pa_high')
            condition = 4;  % high
            return;
        end
    end
    
    % If all else fails
    condition = [];
    runNumber = [];
end

function plotMetricWithBounds(timeSteps, means, uppers, lowers, conditionLabels, colors, shockWin, titleStr, yLabelStr, savePath)
    % Creates a plot showing mean values with upper/lower bounds for each condition
    
    % Create figure
    fig = figure('Color', 'w', 'Position', [100, 100, 1000, 650]);
    ax = axes(fig);
    hold(ax, 'on');
    
    % Plot each condition with bounds
    legendHandles = zeros(1, length(conditionLabels));
    
    for i = 1:length(conditionLabels)
        % Plot mean line
        h = plot(ax, timeSteps, means(i, :), 'LineWidth', 2.5, 'Color', colors(i,:));
        legendHandles(i) = h;
        
        % Plot shaded bounds
        x = [timeSteps, fliplr(timeSteps)];
        y = [uppers(i, :), fliplr(lowers(i, :))];
        fill(ax, x, y, colors(i,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    end
    
    % Mark shock period with vertical lines and shading
    yLim = ylim(ax);
    patch([shockWin(1) shockWin(2) shockWin(2) shockWin(1)], [yLim(1) yLim(1) yLim(2) yLim(2)], ...
        [0.9 0.9 0.9], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    line(ax, [shockWin(1), shockWin(1)], yLim, 'LineStyle', '--', 'Color', 'k', 'LineWidth', 1.5);
    line(ax, [shockWin(2), shockWin(2)], yLim, 'LineStyle', '--', 'Color', 'k', 'LineWidth', 1.5);
    text(ax, mean(shockWin), yLim(2)*0.95, 'Shock Period', ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
         'BackgroundColor', [1 1 1 0.7]);
    
    % Formatting
    title(ax, titleStr, 'FontSize', 16, 'FontWeight', 'bold');
    xlabel(ax, 'Time Step', 'FontSize', 14);
    ylabel(ax, yLabelStr, 'FontSize', 14);
    
    % Make first letter uppercase for legend
    legendLabels = cellfun(@(x) [upper(x(1)) x(2:end)], conditionLabels, 'UniformOutput', false);
    legendLabels = strcat('PA: ', legendLabels);
    
    % Create legend with all place attachment conditions
    leg = legend(ax, legendHandles, legendLabels, 'Location', 'eastoutside', 'FontSize', 12);
    title(leg, 'Place Attachment');
    
    % Adjust figure to accommodate legend
    set(ax, 'Position', [0.1, 0.1, 0.7, 0.8]);
    
    % Add grid
    grid(ax, 'on');
    
    % Save the figure
    saveas(fig, savePath);
    
    % Also save as vector graphics for publication quality
    saveas(fig, [savePath(1:end-4) '.pdf']);
    
    close(fig);
end

function plotPopulationDistribution(popShocked, popNonShocked, conditionLabels, colors, shockStart, shockEnd, savePath)
    % Creates a stacked bar chart showing population distribution before, during, and after shock
    
    % Create figure
    fig = figure('Color', 'w', 'Position', [100, 100, 1200, 800]);
    
    % Select time points to compare: 
    % (1) Pre-shock (before shock starts)
    % (2) During shock (middle of shock period)
    % (3) Post-shock (after shock ends)
    preShockTime = max(1, shockStart - 5);
    duringShockTime = floor((shockStart + shockEnd) / 2);
    postShockTime = min(size(popShocked, 2), shockEnd + 10);
    
    timePoints = [preShockTime, duringShockTime, postShockTime];
    timeLabels = {'Before Shock', 'During Shock', 'After Shock'};
    
    % Set up the bar chart
    numConditions = length(conditionLabels);
    groupWidth = 0.8;
    barWidth = groupWidth / numConditions;
    
    % Create the stacked bar chart
    for t = 1:length(timePoints)
        subplot(1, 3, t);
        
        % Extract data for this time point
        timePoint = timePoints(t);
        
        % Create group positions
        positions = (1:numConditions) - groupWidth/2 + barWidth/2;
        
        % Plot stacked bars for each condition
        for i = 1:numConditions
            % Calculate total population
            totalPop = popShocked(i, timePoint) + popNonShocked(i, timePoint);
            
            % Create stacked bar
            barData = [popNonShocked(i, timePoint), popShocked(i, timePoint)] / totalPop * 100;
            
            % Plot bar
            bar(positions(i), barData, 'stacked', 'FaceColor', 'flat');
            hold on;
            
            % Add percentage text on each section
            text(positions(i), barData(1)/2, sprintf('%.1f%%', barData(1)), ...
                'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', 'k');
            text(positions(i), barData(1) + barData(2)/2, sprintf('%.1f%%', barData(2)), ...
                'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', 'w');
        end
        
        % Set appearance
        title(timeLabels{t}, 'FontSize', 14);
        if t == 1
            ylabel('Population Distribution (%)', 'FontSize', 12);
        end
        
        % Set x-tick labels
        set(gca, 'XTick', positions, 'XTickLabel', strcat('PA: ', cellfun(@(x) [upper(x(1)) x(2:end)], conditionLabels, 'UniformOutput', false)));
        
        % Add legend to first subplot only
        if t == 1
            legend({'Non-Shocked Areas', 'Shocked Areas'}, 'Location', 'southoutside', 'Orientation', 'horizontal');
        end
        
        ylim([0 100]);
        grid on;
    end
    
    % Add overall title
    sgtitle('Population Distribution: Shocked vs. Non-Shocked Areas', 'FontSize', 16, 'FontWeight', 'bold');
    
    % Save the figure
    saveas(fig, savePath);
    
    % Also save as vector graphics for publication quality
    saveas(fig, [savePath(1:end-4) '.pdf']);
    
    close(fig);
end