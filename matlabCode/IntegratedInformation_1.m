% WTT: OK, got it
clear all
close all
clc
%% All the parameters

%##########################################################################
% change it when processing different LMs
LM_FOLDER = 'LW/';
%##########################################################################
BASE_FOLDER = '/Users/tiantong/Desktop/LMR/ICL_LMR/rawData/';
SAVE_BASE_FOLDER = '/Users/tiantong/Desktop/LMR/ICL_LMR/processedData/';

%% Loading all data

%##########################################################################
% iFolder must be edited according to different LMs
for iFolder = 1
%##########################################################################
    foldPath = [BASE_FOLDER, LM_FOLDER, num2str(iFolder)];
    fileFolder = fullfile(foldPath);
    dirOutput = dir(fullfile(fileFolder,'*.txt'));
    % all the txt file names under foldPath, in a cell
    fileNames = {dirOutput.name};

    for jFile=1: length(fileNames)
        cd(foldPath);
        fileName = fileNames(jFile);
        fileName = fileName{1};
        Data = importdata(fileName);
        data = Data.data;
        
        IMU_Thigh=data(:,4:11); % 8 channels, remove yaw
        IMU_Shank=data(:,13:20); % 8 channels, remove yaw
        toe_force=data(:,27);
        heel_force=data(:,29);
        IMU_Thigh(:,1:2)=IMU_Thigh(:,1:2)/10000;
        IMU_Shank(:,1:2)=IMU_Shank(:,1:2)/10000;
        IMU_Thigh_roll=IMU_Thigh(:,2);
        IMU_Shank_roll=IMU_Shank(:,2);
        Knee_angle=IMU_Thigh_roll-IMU_Shank_roll+5; 
        
        saveFoldPath=[SAVE_BASE_FOLDER, LM_FOLDER];
        cd(saveFoldPath);
        save(['converted_',num2str(iFolder),'_',num2str(jFile),'.mat'],'IMU_Thigh','IMU_Shank','Knee_angle','IMU_Thigh_roll','IMU_Shank_roll','toe_force','heel_force');
    end
end
