function runRandomForestAnalysis()

% Load the data with parameter names
dataPath = '/Users/adham/Documents/MIDAS/MIDAS-PA-V3/Outputs/PA-experiments/PA_Sensitivity_Results_Named.mat';
fprintf('Loading data from %s...\n', dataPath);
try
    data = load(dataPath);
    allInputs = data.allInputs;
    allMetrics = data.allMetrics;
    inputNames = data.inputNames;
    metricNames = data.metricNames;
catch ME
    error('Error loading data: %s\nCheck the file path and format.', ME.message);
end

% Verify data integrity
if isempty(allInputs) || isempty(allMetrics)
    error('Input or metrics data is empty. Check the data file.');
end

fprintf('Data loaded successfully.\n');
fprintf('Running Random Forest Analysis with %d samples...\n', size(allInputs, 1));

% Create output directory for plots
outputDir = fullfile(fileparts(dataPath), 'RF_Results');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Settings for Random Forest
numTrees = 500;
minLeafSize = 5;
numPredictorsToSample = 'all'; % Can also use a number like floor(sqrt(size(allInputs,2)))

% Initialize storage for the models and importance scores
rfModels = cell(size(allMetrics, 2), 1);
importanceScores = zeros(size(allInputs, 2), size(allMetrics, 2));
oobErrors = zeros(size(allMetrics, 2), 1);

% Define palette for the parameter types (you can adjust these colors)
paParamsColor = [0.2 0.6 0.8]; % Blue for Place Attachment params
otherParamsColor = [0.8 0.4 0.2]; % Orange for other params

% Only the first 5 parameters are place attachment related
paramColors = repmat(otherParamsColor, size(allInputs, 2), 1);
paramColors(1:5, :) = repmat(paParamsColor, 5, 1);

% Process each metric with Random Forest
for i = 1:size(allMetrics, 2)
    fprintf('Building Random Forest model for %s...\n', metricNames{i});
    
    % Train Random Forest model
    rf = TreeBagger(numTrees, allInputs, allMetrics(:,i), ...
                    'OOBPrediction', 'on', ...
                    'OOBPredictorImportance', 'on', ...
                    'Method', 'regression', ...
                    'MinLeafSize', minLeafSize, ...
                    'NumPredictorsToSample', numPredictorsToSample);
    
    % Store the model
    rfModels{i} = rf;
    
    % Extract and store importance scores
    importanceScores(:, i) = rf.OOBPermutedPredictorDeltaError';
    
    % Try to get OOB error - handle version compatibility
    try
        % Try with newer MATLAB versions
        oobErrors(i) = rf.OOBError(end);
    catch
        % For older MATLAB versions, we'll skip OOB error
        oobErrors(i) = NaN;
        fprintf('Note: OOBError property not available in this MATLAB version.\n');
    end
    
    % Create figure for this metric
    fig = figure('Position', [100 100 1200 800]);
    
    % Sort parameters by importance
    [sortedImportance, sortedIdx] = sort(importanceScores(:, i), 'descend');
    
    % Create bar chart
    barh_handle = barh(sortedImportance);
    
    % Apply colors based on parameter type
    for j = 1:length(sortedIdx)
        barh_handle.FaceColor = 'flat';
        barh_handle.CData(j,:) = paramColors(sortedIdx(j), :);
    end
    
    % Annotations and formatting - handle OOB error display
    if isnan(oobErrors(i))
        title(sprintf('Variable Importance for %s', metricNames{i}), 'FontSize', 14);
    else
        title(sprintf('Variable Importance for %s (OOB Error: %.4f)', metricNames{i}, oobErrors(i)), 'FontSize', 14);
    end
    xlabel('Importance Score (Higher = More Important)', 'FontSize', 12);
    ylabel('Parameters', 'FontSize', 12);
    
    % Use sorted parameter names for y-axis labels
    sortedParamNames = inputNames(sortedIdx);
    set(gca, 'YTick', 1:length(sortedIdx), 'YTickLabel', sortedParamNames, 'FontSize', 10);
    
    % Add a legend
    hold on;
    h1 = plot(NaN, NaN, 'Color', paParamsColor, 'LineWidth', 4);
    h2 = plot(NaN, NaN, 'Color', otherParamsColor, 'LineWidth', 4);
    legend([h1, h2], {'Place Attachment Parameters', 'Other Parameters'}, 'Location', 'SouthEast');
    hold off;
    
    % Add grid lines for better readability
    grid on;
    
    % Adjust figure for better display
    set(gca, 'YDir', 'reverse'); % Reverse order to have highest importance at top
    
    % Save the figure
    saveas(fig, fullfile(outputDir, sprintf('VarImportance_%s.png', metricNames{i})));
    saveas(fig, fullfile(outputDir, sprintf('VarImportance_%s.fig', metricNames{i})));
    
    fprintf('Completed Random Forest analysis for %s.\n', metricNames{i});
end

% Save the models and importance scores
save(fullfile(outputDir, 'RandomForestResults.mat'), 'rfModels', 'importanceScores', 'oobErrors', 'inputNames', 'metricNames');

% Create a summary figure showing top 5 parameters for each metric
figure('Position', [100 100 1500 1000]);

% Determine how many subplots we need (arrange in a grid)
numMetrics = size(allMetrics, 2);
gridSize = ceil(sqrt(numMetrics));
rows = gridSize;
cols = ceil(numMetrics / rows);

% For each metric, plot the top 5 most important parameters
for i = 1:numMetrics
    subplot(rows, cols, i);
    
    % Get top 5 parameters for this metric
    [~, topIdx] = sort(importanceScores(:, i), 'descend');
    topIdx = topIdx(1:min(5, length(topIdx)));
    
    % Create bar chart of top parameters
    barh_handle = barh(importanceScores(topIdx, i));
    
    % Apply colors
    for j = 1:length(topIdx)
        barh_handle.FaceColor = 'flat';
        barh_handle.CData(j,:) = paramColors(topIdx(j), :);
    end
    
    % Labels and formatting
    title(sprintf('%s', metricNames{i}), 'FontSize', 12);
    if i == 1
        ylabel('Importance', 'FontSize', 10);
    end
    
    set(gca, 'YTick', 1:length(topIdx), 'YTickLabel', inputNames(topIdx), 'FontSize', 9);
    set(gca, 'YDir', 'reverse'); % Highest importance at top
    
    % Add grid for readability
    grid on;
end

% Add a super title
sgtitle('Top 5 Most Important Parameters for Each Metric', 'FontSize', 16);

% Save the summary figure
saveas(gcf, fullfile(outputDir, 'Summary_Top5Parameters.png'));
saveas(gcf, fullfile(outputDir, 'Summary_Top5Parameters.fig'));

% Create a heatmap of parameter importance across all metrics
figure('Position', [100 100 1400 800]);

% Normalize importance scores for better visualization
normImportance = zeros(size(importanceScores));
for i = 1:size(importanceScores, 2)
    maxVal = max(importanceScores(:, i));
    if maxVal > 0
        normImportance(:, i) = importanceScores(:, i) / maxVal;
    end
end

% Sort parameters by their average importance across metrics
[~, sortIdx] = sort(mean(normImportance, 2), 'descend');

% Create heatmap
imagesc(normImportance(sortIdx, :)');
colormap('hot');
colorbar;

% Set axis labels and ticks
set(gca, 'XTick', 1:length(sortIdx), 'XTickLabel', inputNames(sortIdx), 'FontSize', 10);
set(gca, 'YTick', 1:numMetrics, 'YTickLabel', metricNames, 'FontSize', 10);
xtickangle(45);

% Add title and labels
title('Relative Importance of Parameters Across All Metrics', 'FontSize', 14);
xlabel('Parameters', 'FontSize', 12);
ylabel('Metrics', 'FontSize', 12);

% Save the heatmap
saveas(gcf, fullfile(outputDir, 'Importance_Heatmap.png'));
saveas(gcf, fullfile(outputDir, 'Importance_Heatmap.fig'));

% Only focus on the place attachment parameters for a specialized analysis
paParamIndices = 1:5; % Assuming first 5 are place attachment parameters

% Create a radar chart for place attachment parameters only
figure('Position', [100 100 1200 800]);

% For each metric, create radar data
for i = 1:numMetrics
    % Normalize PA parameters to range 0-1 for this metric
    paImportance = importanceScores(paParamIndices, i);
    paImportance = paImportance / max(paImportance);
    
    % Plot on polar chart
    subplot(rows, cols, i);
    angles = linspace(0, 2*pi, length(paParamIndices)+1);
    polarplot([angles(1:end-1), angles(1)], [paImportance', paImportance(1)], '-o', 'LineWidth', 2);
    
    % Add title
    title(metricNames{i}, 'FontSize', 12);
    
    % Customize polar chart
    thetaticks(0:72:360);
    thetaticklabels(inputNames(paParamIndices));
    rticks([0 0.25 0.5 0.75 1]);
    rticklabels({'0', '0.25', '0.5', '0.75', '1'});
    grid on;
end

% Save the place attachment radar chart
saveas(gcf, fullfile(outputDir, 'PA_Parameters_Radar.png'));
saveas(gcf, fullfile(outputDir, 'PA_Parameters_Radar.fig'));

fprintf('Analysis complete. Results saved to %s\n', outputDir);
end