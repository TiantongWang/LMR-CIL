% WTT: what's this for? 

clc
clear all
close all
%% All the parameters
LM_FOLDER = 'RA/';
BASE_FOLDER = '/Users/tiantong/Desktop/LMR/ICL_LMR/processedData/';
foldPath=[BASE_FOLDER, LM_FOLDER];
%%
for NO=3:14
    Y=[];
    All_toe_strike=[];
    All_heel_off=[];
    All_toe_off=[];
    for mmmm=1:10
        cd(foldPath);
        load(['Data_',num2str(NO),'_',num2str(mmmm),'.mat']);
        %%
        Time=1:1:length(Knee_angle);
        for i=1:(length(heel_strike)-2)
            x=Time(heel_strike(i):heel_strike(i+1)-1);
            y=Knee_angle(heel_strike(i):heel_strike(i+1)-1,1);
            x_1=linspace(x(1),x(end),100);
            y_1=interp1(x,y,x_1,'spline');
            Y=[Y;y_1];
            
            All_toe_strike=[All_toe_strike;(toe_strike(i)-heel_strike(i))*100/(heel_strike(i+1)-heel_strike(i))];
            All_heel_off=[All_heel_off;(heel_off(i)-heel_strike(i))*100/(heel_strike(i+1)-heel_strike(i))];
            All_toe_off=[All_toe_off;(toe_off(i)-heel_strike(i))*100/(heel_strike(i+1)-heel_strike(i))];
        end
    end
    xx=1:1:100;
    mean_y_1=mean(Y);
    std_y_1=std(Y);
    mean_toe_strike=mean(All_toe_strike);
    std_toe_strike=std(All_toe_strike);
    mean_heel_off=mean(All_heel_off);
    std_heel_off=std(All_heel_off);
    mean_toe_off=mean(All_toe_off);
    std_toe_off=std(All_toe_off);
    cd(foldPath);
%     save(['Gaitdata_',num2str(NO),'.mat'],'mean_y_1','std_y_1','mean_toe_strike','std_toe_strike','xx','mean_heel_off'...
%         ,'std_heel_off','mean_toe_off','std_toe_off','All_toe_strike','All_heel_off','All_toe_off','Y')
end
%%
close all
plot(All_toe_strike,'r')
hold on
plot(All_heel_off,'b')
hold on
plot(All_toe_off,'k')







