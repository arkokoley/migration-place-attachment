function runMIDAS_Full_PA_SA()

clear functions
clear classes

addpath('./Override_Core_MIDAS_Code');
addpath('./Application_Specific_MIDAS_Code');
addpath('./Core_MIDAS_Code');

cd /home/cloud-user/MIDAS-PA-V3

rng('shuffle');

outputList = {};
series = 'PA_SA_';
saveDirectory = './Outputs/';

fprintf(['Building Experiment List.\n']);

experimentList = {};
experiment_table = table([],[],'VariableNames',{'parameterNames','parameterValues'});

for indexI = 1:1000
    experiment = experiment_table;

    owningCost = rand() * 150;
    experiment = [experiment; {'modelParameters.shortName', 'varying_everything'}];
    experiment = [experiment; {'modelParameters.runID', 'VE'}];
    experiment = [experiment; {'modelParameters.placeAttachmentFlag', 1}];
    experiment = [experiment; {'modelParameters.aspirationsFlag', randperm(2,1) - 1}];
    
    % Vary all place attachment parameters
    experiment = [experiment; {'agentParameters.placeAttachmentMean', rand()}];
    experiment = [experiment; {'agentParameters.placeAttachmentSD', rand() * 0.2}];
    experiment = [experiment; {'agentParameters.placeAttachmentGrowMean', rand() * 0.02}];
    experiment = [experiment; {'agentParameters.placeAttachmentGrowSD', rand() * 0.002}];
    experiment = [experiment; {'agentParameters.placeAttachmentDecayMean', rand() * 0.002}];
    experiment = [experiment; {'agentParameters.placeAttachmentDecaySD', rand() * 0.0002}];
    experiment = [experiment; {'agentParameters.initialPlaceAttachmentMean', rand() * 0.7}];
    experiment = [experiment; {'agentParameters.initialPlaceAttachmentSD', rand() * 0.2}];
    
    % Other parameters from original script
    experiment = [experiment; {'agentParameters.rValueMean', rand() * 1.5}];
    experiment = [experiment; {'mapParameters.movingCostsPerMile', rand() * 0.002}];
    experiment = [experiment; {'modelParameters.largeFarmCost', owningCost * 2}];
    experiment = [experiment; {'modelParameters.smallFarmCost', owningCost}];
    experiment = [experiment; {'modelParameters.utility_k', rand() * 4 + 1}];
    experiment = [experiment; {'modelParameters.utility_m', rand() + 1}];
    experiment = [experiment; {'modelParameters.remitRate', rand() * 20}];
    experiment = [experiment; {'mapParameters.minDistForCost', rand() * 50}];
    experiment = [experiment; {'mapParameters.maxDistForCost', rand() * 5000}];
    experiment = [experiment; {'networkParameters.networkDistanceSD', randperm(10,1) + 5}];
    experiment = [experiment; {'networkParameters.connectionsMean', randperm(4,1) + 1}];
    experiment = [experiment; {'networkParameters.connectionsSD', randperm(2,1) + 1}];
    experiment = [experiment; {'networkParameters.weightLocation', rand() * 10 + 5}];
    experiment = [experiment; {'networkParameters.weightNetworkLink', rand() * 10 + 5}];
    experiment = [experiment; {'networkParameters.weightSameLayer', rand() * 7 + 3}];
    experiment = [experiment; {'networkParameters.distancePolynomial', rand() * 0.0002 + 0.0001}];
    experiment = [experiment; {'networkParameters.decayPerStep', max(0.001, rand() * 0.01)}];
    experiment = [experiment; {'networkParameters.interactBump', max(0.005, rand() * 0.03)}];
    experiment = [experiment; {'networkParameters.shareBump', max(0.0005, rand() * 0.005)}];
    
    experimentList{end+1} = experiment;
end

experimentList = experimentList(randperm(length(experimentList)));

fprintf(['Saving Experiment List.\n']);
save([saveDirectory 'PA_FS_' date '_input_summary'], 'experimentList');

numRuns = length(experimentList);
runList = zeros(length(experimentList),1);

parfor indexI = 1:length(experimentList)
    if(runList(indexI) == 0)
        input = experimentList{indexI};
        output = midasMainLoop(input, ['Experiment Run ' num2str(indexI) ' of ' num2str(numRuns)]);
        functionVersions = inmem('-completenames');
        functionVersions = functionVersions(strmatch(pwd,functionVersions));
        output.codeUsed = functionVersions;
        currentFile = [series num2str(length(dir([series '*']))) '_' datestr(now) '.mat'];
        currentFile = [saveDirectory currentFile];
        currentFile = strrep(currentFile, ':', '-');
        currentFile = strrep(currentFile, ' ', '_');
        saveToFile(input, output, currentFile);
        runList(indexI) = 1;
    end
end

end

function saveToFile(input, output, filename)
save(filename,'input', 'output');
end
