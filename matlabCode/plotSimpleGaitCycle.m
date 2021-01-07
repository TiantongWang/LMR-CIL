% Only use heel strikes to divide signals into gait cycles
% didn't use other critical moments

clc
clear all
close all

%% All the parameters
BASE_FOLDER = '/Users/tiantong/Desktop/LMR/ICL_LMR/processedData/';
RA_FOLDER = 'RA/';
RD_FOLDER = 'RD/';
LW_FOLDER = 'LW/';

%##########################################################################
% edit it when you want to see other angles
RA_DEGREE = [5, 10, 14];
RD_DEGREE = [5, 10, 14];
LW_DEGREE = [1]; % ummm, weird
%##########################################################################

%% plot
figure();
% load and plot RA
foldPath=[BASE_FOLDER, RA_FOLDER];
cd(foldPath);
for iFile = 1: length(RA_DEGREE)
    load(['Gaitdata_',num2str(RA_DEGREE(iFile)),'.mat']);
    plot(xx, mean_y_1, 'Linewidth', 2);
%     legend(['RA', ' ', num2str(RA_DEGREE(iFile))]);
    hold on;
end

% load and plot RD
foldPath=[BASE_FOLDER, RD_FOLDER];
cd(foldPath);
for iFile = 1: length(RD_DEGREE)
    load(['Gaitdata_',num2str(RD_DEGREE(iFile)),'.mat']);
    plot(xx, mean_y_1, 'Linewidth', 2);
%     legend(['RA', ' ', num2str(RA_DEGREE(iFile))]);
    hold on;
end

% load and plot LW
foldPath=[BASE_FOLDER, LW_FOLDER];
cd(foldPath);
for iFile = 1: length(LW_DEGREE)
    load(['Gaitdata_',num2str(LW_DEGREE(iFile)),'.mat']);
    plot(xx, mean_y_1, 'Linewidth', 2);
%     legend(['RA', ' ', num2str(RA_DEGREE(iFile))]);
    hold on;
end
            
