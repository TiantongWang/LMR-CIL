% Only use heel strikes to divide signals into gait cycles
% didn't use other critical moments

clc
clear all
close all
%% All the parameters
%##########################################################################
% change it when processing different LMs
LM_FOLDER = 'LW/';
%##########################################################################
BASE_FOLDER = '/Users/tiantong/Desktop/LMR/ICL_LMR/processedData/';
foldPath=[BASE_FOLDER, LM_FOLDER];
%##########################################################################
% if the number of files in a folder changes, edit it!
nFilePerFolder = 10;
%##########################################################################
%%
%##########################################################################
% iFolder must be edited according to different LMs
for iFolder=1
%##########################################################################
    Y=[];
%     All_toe_strike=[];
%     All_heel_off=[];
%     All_toe_off=[];
    for jFile = 1: nFilePerFolder
        cd(foldPath);
        load(['Data_',num2str(iFolder),'_',num2str(jFile),'.mat']);
        %%
        Time=1:1:length(Knee_angle); % time tick
        for i=1:(length(heel_strike)-1)
            x=Time(heel_strike(i):heel_strike(i+1)-1);
            y=Knee_angle(heel_strike(i):heel_strike(i+1)-1,1);
            x_1=linspace(x(1),x(end),100);
            y_1=interp1(x,y,x_1,'spline');
            Y=[Y;y_1];
            
%             All_toe_strike=[All_toe_strike;(toe_strike(i)-heel_strike(i))*100/(heel_strike(i+1)-heel_strike(i))];
%             All_heel_off=[All_heel_off;(heel_off(i)-heel_strike(i))*100/(heel_strike(i+1)-heel_strike(i))];
%             All_toe_off=[All_toe_off;(toe_off(i)-heel_strike(i))*100/(heel_strike(i+1)-heel_strike(i))];
        end
    end
    xx=1:1:100;
    mean_y_1=mean(Y);
    std_y_1=std(Y);
%     mean_toe_strike=mean(All_toe_strike);
%     std_toe_strike=std(All_toe_strike);
%     mean_heel_off=mean(All_heel_off);
%     std_heel_off=std(All_heel_off);
%     mean_toe_off=mean(All_toe_off);
%     std_toe_off=std(All_toe_off);
    cd(foldPath);
%     save(['Gaitdata_',num2str(NO),'.mat'],'mean_y_1','std_y_1','mean_toe_strike','std_toe_strike','xx','mean_heel_off'...
%         ,'std_heel_off','mean_toe_off','std_toe_off','All_toe_strike','All_heel_off','All_toe_off','Y')
    save(['Gaitdata_',num2str(iFolder),'.mat'],'mean_y_1','std_y_1','xx','Y');
end
%%
% close all
% plot(All_toe_strike,'r')
% hold on
% plot(All_heel_off,'b')
% hold on
% plot(All_toe_off,'k')

