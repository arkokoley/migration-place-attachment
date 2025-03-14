function analyzePlaceAttachmentSensitivity()

experimentDir = '/Users/adham/Documents/MIDAS/MIDAS-PA-V3/Outputs';
saveDir = '/Users/adham/Documents/MIDAS/MIDAS-PA-V3/Outputs/PA-experiments';

outputFiles = dir(fullfile(experimentDir, 'PA_SA*.mat'));

% Initialize arrays for storing results
allMetrics = [];
allInputs = [];

numFiles = length(outputFiles);
fprintf('Processing %d files...\n', numFiles);
tic; % Start a timer to track progress

for k = 1:numFiles
    filePath = fullfile(outputFiles(k).folder, outputFiles(k).name);
    try
        % Check if file exists
        if ~exist(filePath, 'file')
            fprintf('Warning: File %s does not exist. Skipping...\n', outputFiles(k).name);
            continue;
        end
        
        % Try to load the file
        data = load(filePath);
        
    catch ME
        fprintf('Error loading file %s: %s\nSkipping this file...\n', outputFiles(k).name, ME.message);
        continue;
    end
    
    if isfield(data, 'output')
        % Extract output metrics
        numMigrations = sum(data.output.migrations(:));
        avgWealth = mean(data.output.averageWealth(:));
        numAgents = length(data.output.agentSummary.id);
        numLocations = size(data.output.mapVariables.locations, 1);
        distanceMatrix = data.output.mapVariables.distanceMatrix_scaled;
        
        % Extract movement-related metrics
        agentSummary = data.output.agentSummary;
        numReturns = zeros(numAgents, 1);
        distanceTraveled = zeros(numAgents, 1);
        totalMoves = zeros(numAgents, 1);
        avgDistancePerMove = zeros(numAgents, 1);
        fractionMovesHome = zeros(numAgents, 1);
        avgDistanceAwayFromHome = zeros(numAgents, 1);
        timeStepsAwayFromHome = zeros(numAgents, 1);
        
        for i = 1:numAgents
            moveHistory = agentSummary.moveHistory{i};
            if ~isempty(moveHistory) && ismatrix(moveHistory) && size(moveHistory, 2) >= 2
                initialLocation = moveHistory(1, 2);
                returnsToHome = sum(moveHistory(:, 2) == initialLocation);
                numReturns(i) = returnsToHome - 1;
                totalMoves(i) = size(moveHistory, 1) - 1;
                totalDistanceAway = 0;
                countMovesAwayFromHome = 0;
                
                for j = 2:size(moveHistory, 1)
                    fromLocation = moveHistory(j - 1, 2);
                    toLocation = moveHistory(j, 2);
                    distanceTraveled(i) = distanceTraveled(i) + distanceMatrix(fromLocation, toLocation);
                    if toLocation ~= initialLocation
                        totalDistanceAway = totalDistanceAway + distanceMatrix(initialLocation, toLocation);
                        countMovesAwayFromHome = countMovesAwayFromHome + 1;
                    end
                end
                
                if totalMoves(i) > 0
                    avgDistancePerMove(i) = distanceTraveled(i) / totalMoves(i);
                    fractionMovesHome(i) = numReturns(i) / totalMoves(i);
                else
                    avgDistancePerMove(i) = 0;
                    fractionMovesHome(i) = 0;
                end
                
                if countMovesAwayFromHome > 0
                    avgDistanceAwayFromHome(i) = totalDistanceAway / countMovesAwayFromHome;
                else
                    avgDistanceAwayFromHome(i) = 0;
                end
                
                timeStepsAwayFromHome(i) = countMovesAwayFromHome;
            end
        end
        
        % Extract input parameters
        inputData = load(filePath, 'input');
        parameterNames = inputData.input.parameterNames;
        parameterValues = inputData.input.parameterValues;
        
        % Create a row vector with each parameter as its own element
        inputParams = zeros(1, 24);  % Initialize with the correct number of parameters
        
        % Extract each parameter individually and ensure it's stored as a single element in a row
        inputParams(1) = parameterValues{strcmp(parameterNames, 'agentParameters.placeAttachmentMean')};
        inputParams(2) = parameterValues{strcmp(parameterNames, 'agentParameters.placeAttachmentSD')};
        inputParams(3) = parameterValues{strcmp(parameterNames, 'agentParameters.placeAttachmentGrowMean')};
        inputParams(4) = parameterValues{strcmp(parameterNames, 'agentParameters.placeAttachmentDecayMean')};
        inputParams(5) = parameterValues{strcmp(parameterNames, 'agentParameters.initialPlaceAttachmentMean')};
        inputParams(6) = parameterValues{strcmp(parameterNames, 'agentParameters.rValueMean')};
        inputParams(7) = parameterValues{strcmp(parameterNames, 'mapParameters.movingCostsPerMile')};
        inputParams(8) = parameterValues{strcmp(parameterNames, 'modelParameters.largeFarmCost')};
        inputParams(9) = parameterValues{strcmp(parameterNames, 'modelParameters.smallFarmCost')};
        inputParams(10) = parameterValues{strcmp(parameterNames, 'modelParameters.utility_k')};
        inputParams(11) = parameterValues{strcmp(parameterNames, 'modelParameters.utility_m')};
        inputParams(12) = parameterValues{strcmp(parameterNames, 'modelParameters.remitRate')};
        inputParams(13) = parameterValues{strcmp(parameterNames, 'mapParameters.minDistForCost')};
        inputParams(14) = parameterValues{strcmp(parameterNames, 'mapParameters.maxDistForCost')};
        inputParams(15) = parameterValues{strcmp(parameterNames, 'networkParameters.networkDistanceSD')};
        inputParams(16) = parameterValues{strcmp(parameterNames, 'networkParameters.connectionsMean')};
        inputParams(17) = parameterValues{strcmp(parameterNames, 'networkParameters.connectionsSD')};
        inputParams(18) = parameterValues{strcmp(parameterNames, 'networkParameters.weightLocation')};
        inputParams(19) = parameterValues{strcmp(parameterNames, 'networkParameters.weightNetworkLink')};
        inputParams(20) = parameterValues{strcmp(parameterNames, 'networkParameters.weightSameLayer')};
        inputParams(21) = parameterValues{strcmp(parameterNames, 'networkParameters.distancePolynomial')};
        inputParams(22) = parameterValues{strcmp(parameterNames, 'networkParameters.decayPerStep')};
        inputParams(23) = parameterValues{strcmp(parameterNames, 'networkParameters.interactBump')};
        inputParams(24) = parameterValues{strcmp(parameterNames, 'networkParameters.shareBump')};
        
        % Store results
        allMetrics = [allMetrics; numMigrations, avgWealth, numAgents, numLocations, mean(numReturns), mean(distanceTraveled), mean(totalMoves), mean(avgDistancePerMove), mean(fractionMovesHome), mean(avgDistanceAwayFromHome), mean(timeStepsAwayFromHome)];
        allInputs = [allInputs; inputParams];
    end

    % Print progress every 10 files
    if mod(k, 10) == 0 || k == numFiles
        elapsedTime = toc;
        avgTimePerFile = elapsedTime / k;
        remainingTime = avgTimePerFile * (numFiles - k);
        fprintf('Processed %d/%d files. Estimated time remaining: %.2f minutes.\n', k, numFiles, remainingTime / 60);
    end
end

% Check if we collected any data
if isempty(allMetrics)
    fprintf('Warning: No valid data was collected. Check your files and paths.\n');
    return;
end

% Save intermediate results
save(fullfile(saveDir, 'PA_Sensitivity_Results.mat'), 'allMetrics', 'allInputs');
fprintf('All results saved.\n');

% Add column names for better interpretability
metricNames = {'numMigrations', 'avgWealth', 'numAgents', 'numLocations', 'avgNumReturns', ...
               'avgDistanceTraveled', 'avgTotalMoves', 'avgDistancePerMove', 'avgFractionMovesHome', ...
               'avgDistanceAwayFromHome', 'avgTimeStepsAwayFromHome'};
           
inputNames = {'placeAttachmentMean', 'placeAttachmentSD', 'placeAttachmentGrowMean', ...
              'placeAttachmentDecayMean', 'initialPlaceAttachmentMean', 'rValueMean', ...
              'movingCostsPerMile', 'largeFarmCost', 'smallFarmCost', 'utility_k', ...
              'utility_m', 'remitRate', 'minDistForCost', 'maxDistForCost', ...
              'networkDistanceSD', 'connectionsMean', 'connectionsSD', 'weightLocation', ...
              'weightNetworkLink', 'weightSameLayer', 'distancePolynomial', ...
              'decayPerStep', 'interactBump', 'shareBump'};
          
% Save with column names for easier analysis
save(fullfile(saveDir, 'PA_Sensitivity_Results_Named.mat'), 'allMetrics', 'allInputs', 'metricNames', 'inputNames');
fprintf('Results with column names saved.\n');

end