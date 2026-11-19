function body_analysis(outputFile)
% BODY_ANALYSIS
% Calculates body-segment lengths, volumes, and masses from point-cloud
% data, then exports the results to a text file for the Python/OpenSim
% automation pipeline.
%
% Run normally from MATLAB:
%   body_analysis
%
% Run with a custom output location:
%   body_analysis('C:\CapstonePipeline\output\subject_measurements.txt')
%
% Run from Python/MATLAB batch mode:
%   matlab -batch "body_analysis('subject_measurements.txt')"

%% --- INITIAL SETUP ---

clc;
close all;

disp('=============================================');
disp('     STARTING BODY ANALYSIS PIPELINE         ');
disp('=============================================');

% Find the folder containing this MATLAB file.
scriptPath = mfilename('fullpath');
scriptFolder = fileparts(scriptPath);

% If no output filename was supplied, save the text file beside this script.
if nargin < 1 || isempty(outputFile)
    outputFile = fullfile(scriptFolder, 'subject_measurements.txt');
end

% Make sure the output path is absolute.
if ~isAbsolutePath(outputFile)
    outputFile = fullfile(scriptFolder, outputFile);
end

% Create the output folder if it does not exist.
outputFolder = fileparts(outputFile);

if ~isempty(outputFolder) && ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

fprintf('MATLAB script folder: %s\n', scriptFolder);
fprintf('Output file: %s\n\n', outputFile);

%% --- LOAD POINT CLOUD AND SEGMENT DATA ---

disp('Loading point-cloud and segment files...');

pointsFile = fullfile(scriptFolder, 'points.mat');
facesFile = fullfile(scriptFolder, 'facesShapeModel.mat');
idxLAFile = fullfile(scriptFolder, 'idxLA.mat');
idxLLFile = fullfile(scriptFolder, 'idxLL.mat');
idxRAFile = fullfile(scriptFolder, 'idxRA.mat');
idxRLFile = fullfile(scriptFolder, 'idxRL.mat');
totalIdxFile = fullfile(scriptFolder, 'TotalIdx.mat');
idxHeadFile = fullfile(scriptFolder, 'idxHead.mat');

requiredFiles = {
    pointsFile
    facesFile
    idxLAFile
    idxLLFile
    idxRAFile
    idxRLFile
    totalIdxFile
    idxHeadFile
};

% Confirm every required file exists before continuing.
for fileNumber = 1:length(requiredFiles)
    if ~isfile(requiredFiles{fileNumber})
        error('Required file was not found: %s', requiredFiles{fileNumber});
    end
end

% Load each file into a structure.
pointsData = load(pointsFile);
facesData = load(facesFile);
idxLAData = load(idxLAFile);
idxLLData = load(idxLLFile);
idxRAData = load(idxRAFile);
idxRLData = load(idxRLFile);
totalIdxData = load(totalIdxFile);
idxHeadData = load(idxHeadFile);

% Extract the required variables.
points = pointsData.points;
faces = facesData.faces;
idxLA = idxLAData.idxLA;
idxLL = idxLLData.idxLL;
idxRA = idxRAData.idxRA;
idxRL = idxRLData.idxRL;
TotalIdx = totalIdxData.TotalIdx;
idxHead = idxHeadData.idxHead;

disp('All point-cloud files loaded successfully.');

%% --- VALIDATE INPUT DATA ---

if size(points, 2) ~= 3
    error('The points variable must contain three columns: X, Y, and Z.');
end

if size(faces, 2) ~= 3
    error('The faces variable must contain three vertex indices per row.');
end

if any(~isfinite(points), 'all')
    error('The point cloud contains NaN or Inf values.');
end

%% --- VISUALIZATION OF THE ENTIRE BODY ---

figure( ...
    'Name', 'Entire Body Shape', ...
    'NumberTitle', 'off' ...
);

patch( ...
    'Faces', faces, ...
    'Vertices', points, ...
    'FaceColor', [22, 111, 245] / 255, ...
    'EdgeColor', 'black' ...
);

axis equal;
xlabel('X (mm)');
ylabel('Y (mm)');
zlabel('Z (mm)');
title('Entire Body Point-Cloud Model');
view(-30, 22.5);
grid on;

%% --- VISUALIZATION OF SEPARATED BODY SEGMENTS ---

figure( ...
    'Name', 'Body Shape by Segment', ...
    'NumberTitle', 'off' ...
);

hold on;

patch( ...
    'Faces', faces(idxLL, :), ...
    'Vertices', points, ...
    'FaceColor', [22, 111, 245] / 255, ...
    'EdgeColor', 'black' ...
);

patch( ...
    'Faces', faces(idxRL, :), ...
    'Vertices', points, ...
    'FaceColor', [22, 111, 245] / 255, ...
    'EdgeColor', 'black' ...
);

patch( ...
    'Faces', faces(idxHead, :), ...
    'Vertices', points, ...
    'FaceColor', [22, 208, 245] / 255, ...
    'EdgeColor', 'black' ...
);

patch( ...
    'Faces', faces(idxLA, :), ...
    'Vertices', points, ...
    'FaceColor', [56, 143, 161] / 255, ...
    'EdgeColor', 'black' ...
);

patch( ...
    'Faces', faces(idxRA, :), ...
    'Vertices', points, ...
    'FaceColor', [56, 143, 161] / 255, ...
    'EdgeColor', 'black' ...
);

patch( ...
    'Faces', faces(TotalIdx, :), ...
    'Vertices', points, ...
    'FaceColor', [79, 232, 139] / 255, ...
    'EdgeColor', 'black' ...
);

hold off;

axis equal;
xlabel('X (mm)');
ylabel('Y (mm)');
zlabel('Z (mm)');
title('Body Shape Separated by Segment');
view(-30, 22.5);
grid on;

%% --- DEFINE TISSUE DENSITY ---

% The point-cloud coordinates are assumed to be in millimetres.
%
% 1 mm^3 = 1e-9 m^3
% Approximate tissue density = 1000 kg/m^3
%
% Therefore:
% 1000 kg/m^3 * 1e-9 m^3/mm^3 = 1e-6 kg/mm^3

Density = 1e-6;

%% --- CALCULATE LEFT ARM LENGTH, VOLUME, AND MASS ---

LA_indices = unique(faces(idxLA, :));
LA_points = points(LA_indices, :);

Length_LA = max(LA_points(:, 3)) - min(LA_points(:, 3));

[~, Volume_LA] = convhull( ...
    LA_points(:, 1), ...
    LA_points(:, 2), ...
    LA_points(:, 3) ...
);

Mass_LA = Volume_LA * Density;

%% --- CALCULATE RIGHT ARM LENGTH, VOLUME, AND MASS ---

RA_indices = unique(faces(idxRA, :));
RA_points = points(RA_indices, :);

Length_RA = max(RA_points(:, 3)) - min(RA_points(:, 3));

[~, Volume_RA] = convhull( ...
    RA_points(:, 1), ...
    RA_points(:, 2), ...
    RA_points(:, 3) ...
);

Mass_RA = Volume_RA * Density;

%% --- CALCULATE LEFT LEG LENGTH, VOLUME, AND MASS ---

LL_indices = unique(faces(idxLL, :));
LL_points = points(LL_indices, :);

Length_LL = max(LL_points(:, 3)) - min(LL_points(:, 3));

[~, Volume_LL] = convhull( ...
    LL_points(:, 1), ...
    LL_points(:, 2), ...
    LL_points(:, 3) ...
);

Mass_LL = Volume_LL * Density;

%% --- CALCULATE RIGHT LEG LENGTH, VOLUME, AND MASS ---

RL_indices = unique(faces(idxRL, :));
RL_points = points(RL_indices, :);

Length_RL = max(RL_points(:, 3)) - min(RL_points(:, 3));

[~, Volume_RL] = convhull( ...
    RL_points(:, 1), ...
    RL_points(:, 2), ...
    RL_points(:, 3) ...
);

Mass_RL = Volume_RL * Density;

%% --- CALCULATE HEAD LENGTH, VOLUME, AND MASS ---

Head_indices = unique(faces(idxHead, :));
Head_points = points(Head_indices, :);

Length_Head = max(Head_points(:, 3)) - min(Head_points(:, 3));

[~, Volume_Head] = convhull( ...
    Head_points(:, 1), ...
    Head_points(:, 2), ...
    Head_points(:, 3) ...
);

Mass_Head = Volume_Head * Density;

%% --- CALCULATE TRUNK LENGTH, VOLUME, AND MASS ---

Trunk_indices = unique(faces(TotalIdx, :));
Trunk_points = points(Trunk_indices, :);

Length_Trunk = max(Trunk_points(:, 3)) - min(Trunk_points(:, 3));

[~, Volume_Trunk] = convhull( ...
    Trunk_points(:, 1), ...
    Trunk_points(:, 2), ...
    Trunk_points(:, 3) ...
);

Mass_Trunk = Volume_Trunk * Density;

%% --- CALCULATE TOTAL BODY MASS AND VOLUME ---

Total_Mass = ...
    Mass_LA + ...
    Mass_RA + ...
    Mass_LL + ...
    Mass_RL + ...
    Mass_Head + ...
    Mass_Trunk;

Total_Volume = ...
    Volume_LA + ...
    Volume_RA + ...
    Volume_LL + ...
    Volume_RL + ...
    Volume_Head + ...
    Volume_Trunk;

%% --- DISPLAY RESULTS IN MATLAB COMMAND WINDOW ---

disp(' ');
disp('=============================================');
disp('          SEGMENT MASS DISTRIBUTION          ');
disp('=============================================');

fprintf( ...
    'Left Arm:   %6.2f kg | Length: %6.1f mm | Volume: %.2f mm^3\n', ...
    Mass_LA, Length_LA, Volume_LA ...
);

fprintf( ...
    'Right Arm:  %6.2f kg | Length: %6.1f mm | Volume: %.2f mm^3\n', ...
    Mass_RA, Length_RA, Volume_RA ...
);

fprintf( ...
    'Left Leg:   %6.2f kg | Length: %6.1f mm | Volume: %.2f mm^3\n', ...
    Mass_LL, Length_LL, Volume_LL ...
);

fprintf( ...
    'Right Leg:  %6.2f kg | Length: %6.1f mm | Volume: %.2f mm^3\n', ...
    Mass_RL, Length_RL, Volume_RL ...
);

fprintf( ...
    'Head:       %6.2f kg | Length: %6.1f mm | Volume: %.2f mm^3\n', ...
    Mass_Head, Length_Head, Volume_Head ...
);

fprintf( ...
    'Trunk:      %6.2f kg | Length: %6.1f mm | Volume: %.2f mm^3\n', ...
    Mass_Trunk, Length_Trunk, Volume_Trunk ...
);

disp('---------------------------------------------');
fprintf('TOTAL VOLUME: %.2f mm^3\n', Total_Volume);
fprintf('TOTAL MASS:   %.2f kg\n', Total_Mass);
disp('=============================================');

%% --- EXPORT RESULTS TO TEXT FILE FOR PYTHON ---

fileID = fopen(outputFile, 'w');

if fileID == -1
    error('MATLAB could not create the output file: %s', outputFile);
end

% Write pipeline metadata.
fprintf(fileID, 'STATUS=SUCCESS\n');
fprintf(fileID, 'DENSITY_KG_PER_MM3=%.10f\n', Density);

% Write total measurements first.
fprintf(fileID, 'TOTAL_MASS=%.6f\n', Total_Mass);
fprintf(fileID, 'TOTAL_VOLUME=%.6f\n', Total_Volume);

% Write segment masses.
fprintf(fileID, 'LEFT_ARM_MASS=%.6f\n', Mass_LA);
fprintf(fileID, 'RIGHT_ARM_MASS=%.6f\n', Mass_RA);
fprintf(fileID, 'LEFT_LEG_MASS=%.6f\n', Mass_LL);
fprintf(fileID, 'RIGHT_LEG_MASS=%.6f\n', Mass_RL);
fprintf(fileID, 'HEAD_MASS=%.6f\n', Mass_Head);
fprintf(fileID, 'TRUNK_MASS=%.6f\n', Mass_Trunk);

% Write segment lengths.
fprintf(fileID, 'LEFT_ARM_LENGTH=%.6f\n', Length_LA);
fprintf(fileID, 'RIGHT_ARM_LENGTH=%.6f\n', Length_RA);
fprintf(fileID, 'LEFT_LEG_LENGTH=%.6f\n', Length_LL);
fprintf(fileID, 'RIGHT_LEG_LENGTH=%.6f\n', Length_RL);
fprintf(fileID, 'HEAD_LENGTH=%.6f\n', Length_Head);
fprintf(fileID, 'TRUNK_LENGTH=%.6f\n', Length_Trunk);

% Write segment volumes.
fprintf(fileID, 'LEFT_ARM_VOLUME=%.6f\n', Volume_LA);
fprintf(fileID, 'RIGHT_ARM_VOLUME=%.6f\n', Volume_RA);
fprintf(fileID, 'LEFT_LEG_VOLUME=%.6f\n', Volume_LL);
fprintf(fileID, 'RIGHT_LEG_VOLUME=%.6f\n', Volume_RL);
fprintf(fileID, 'HEAD_VOLUME=%.6f\n', Volume_Head);
fprintf(fileID, 'TRUNK_VOLUME=%.6f\n', Volume_Trunk);

% Close the output file.
closeStatus = fclose(fileID);

if closeStatus ~= 0
    error('MATLAB could not close the output file correctly.');
end

%% --- CONFIRM OUTPUT ---

if ~isfile(outputFile)
    error('The measurement file was not created successfully.');
end

disp(' ');
disp('MATLAB analysis completed successfully.');
fprintf('Measurement file created:\n%s\n', outputFile);
disp('=============================================');

end

%% --- LOCAL HELPER FUNCTION ---

function result = isAbsolutePath(filePath)
% Determines whether a Windows, macOS, or Linux path is absolute.

    if ispc
        result = ...
            ~isempty(regexp(filePath, '^[A-Za-z]:[\\/]', 'once')) || ...
            startsWith(filePath, '\\');
    else
        result = startsWith(filePath, '/');
    end
end