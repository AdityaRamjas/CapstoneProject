%% 
clear all
clc
close all

%% Load point clouds and segment Id
% cd('C:\MATLAB\BodyShape') % change this path to your path
load('points.mat') % load point clouds
load("facesShapeModel.mat"); % load triangulation 
load('idxLA.mat') % load index for left arm (LA)
load('idxLL.mat') % load index for left leg (LL)
load('idxRA.mat') % load index for right arm (RA)
load('idxRL.mat') % load index for right leg (RL)
load('TotalIdx.mat') % load index for the entire trunk
load('idxHead.mat') % load index for the head

%% Visualization of the entire body
figure("Name","Entire Body shape")
patch('Faces',faces,'Vertices',points,'FaceColor',[22,111,245]/255,'Edgecolor','black');axis equal; view(-30,22.5);

%% Visualization of body shape with separated body segments
figure("Name","Body shape by parts")
patch('Faces',faces(idxLL,:),'Vertices',points,'FaceColor',[22,111,245]/255,'Edgecolor','black');axis equal; view(-30,22.5);
patch('Faces',faces(idxRL,:),'Vertices',points,'FaceColor',[22,111,245]/255,'Edgecolor','black');axis equal; view(-30,22.5);
patch('Faces',faces(idxHead,:),'Vertices',points,'FaceColor',[22,208,245]/255,'Edgecolor','black');axis equal; view(-30,22.5);
patch('Faces',faces(idxLA,:),'Vertices',points,'FaceColor',[56,143,161]/255,'Edgecolor','black');axis equal; view(-30,22.5);
patch('Faces',faces(idxRA,:),'Vertices',points,'FaceColor',[56,143,161]/255,'Edgecolor','black');axis equal; view(-30,22.5);
patch('Faces',faces(TotalIdx,:),'Vertices',points,'FaceColor',[79,232,139]/255,'Edgecolor','black');axis equal; view(-30,22.5);

%% --- CALCULATING LENGTH, VOLUME, AND MASS FOR ALL SEGMENTS ---

% Define tissue density (assuming points are in mm, 1e-6 kg/mm^3 = ~1000 kg/m^3)
Density = 1e-6; 

% --- 1. LEFT ARM ---
LA_indices = unique(faces(idxLA, :));
LA_points = points(LA_indices, :);
Length_LA = max(LA_points(:, 3)) - min(LA_points(:, 3));
[~, Volume_LA] = convhull(LA_points(:, 1), LA_points(:, 2), LA_points(:, 3));
Mass_LA = Volume_LA * Density;

% --- 2. RIGHT ARM ---
RA_indices = unique(faces(idxRA, :));
RA_points = points(RA_indices, :);
Length_RA = max(RA_points(:, 3)) - min(RA_points(:, 3));
[~, Volume_RA] = convhull(RA_points(:, 1), RA_points(:, 2), RA_points(:, 3));
Mass_RA = Volume_RA * Density;

% --- 3. LEFT LEG ---
LL_indices = unique(faces(idxLL, :));
LL_points = points(LL_indices, :);
Length_LL = max(LL_points(:, 3)) - min(LL_points(:, 3));
[~, Volume_LL] = convhull(LL_points(:, 1), LL_points(:, 2), LL_points(:, 3));
Mass_LL = Volume_LL * Density;

% --- 4. RIGHT LEG ---
RL_indices = unique(faces(idxRL, :));
RL_points = points(RL_indices, :);
Length_RL = max(RL_points(:, 3)) - min(RL_points(:, 3));
[~, Volume_RL] = convhull(RL_points(:, 1), RL_points(:, 2), RL_points(:, 3));
Mass_RL = Volume_RL * Density;

% --- 5. HEAD ---
Head_indices = unique(faces(idxHead, :));
Head_points = points(Head_indices, :);
Length_Head = max(Head_points(:, 3)) - min(Head_points(:, 3));
[~, Volume_Head] = convhull(Head_points(:, 1), Head_points(:, 2), Head_points(:, 3));
Mass_Head = Volume_Head * Density;

% --- 6. TRUNK ---
Trunk_indices = unique(faces(TotalIdx, :));
Trunk_points = points(Trunk_indices, :);
Length_Trunk = max(Trunk_points(:, 3)) - min(Trunk_points(:, 3));
[~, Volume_Trunk] = convhull(Trunk_points(:, 1), Trunk_points(:, 2), Trunk_points(:, 3));
Mass_Trunk = Volume_Trunk * Density;

% --- 7. TOTAL BODY MASS ---
Total_Mass = Mass_LA + Mass_RA + Mass_LL + Mass_RL + Mass_Head + Mass_Trunk;

%% --- DISPLAY RESULTS IN COMMAND WINDOW ---
disp('=============================================');
disp('          SEGMENT MASS DISTRIBUTION          ');
disp('=============================================');
fprintf('Left Arm:   %6.2f kg  | Length: %6.1f mm\n', Mass_LA, Length_LA);
fprintf('Right Arm:  %6.2f kg  | Length: %6.1f mm\n', Mass_RA, Length_RA);
fprintf('Left Leg:   %6.2f kg  | Length: %6.1f mm\n', Mass_LL, Length_LL);
fprintf('Right Leg:  %6.2f kg  | Length: %6.1f mm\n', Mass_RL, Length_RL);
fprintf('Head:       %6.2f kg  | Length: %6.1f mm\n', Mass_Head, Length_Head);
fprintf('Trunk:      %6.2f kg  | Length: %6.1f mm\n', Mass_Trunk, Length_Trunk);
disp('---------------------------------------------');
fprintf('TOTAL MASS: %6.2f kg\n', Total_Mass);
disp('=============================================');
%% --- EXPORT RESULTS FOR PYTHON ---

fid = fopen('subject_measurements.txt', 'w');

fprintf(fid, 'TOTAL_MASS=%.2f\n', Total_Mass);

fprintf(fid, 'LEFT_ARM_MASS=%.2f\n', Mass_LA);
fprintf(fid, 'RIGHT_ARM_MASS=%.2f\n', Mass_RA);

fprintf(fid, 'LEFT_LEG_MASS=%.2f\n', Mass_LL);
fprintf(fid, 'RIGHT_LEG_MASS=%.2f\n', Mass_RL);

fprintf(fid, 'HEAD_MASS=%.2f\n', Mass_Head);
fprintf(fid, 'TRUNK_MASS=%.2f\n', Mass_Trunk);

fprintf(fid, 'LEFT_ARM_LENGTH=%.2f\n', Length_LA);
fprintf(fid, 'RIGHT_ARM_LENGTH=%.2f\n', Length_RA);

fprintf(fid, 'LEFT_LEG_LENGTH=%.2f\n', Length_LL);
fprintf(fid, 'RIGHT_LEG_LENGTH=%.2f\n', Length_RL);

fprintf(fid, 'HEAD_LENGTH=%.2f\n', Length_Head);
fprintf(fid, 'TRUNK_LENGTH=%.2f\n', Length_Trunk);

fclose(fid);

disp('subject_measurements.txt generated successfully');