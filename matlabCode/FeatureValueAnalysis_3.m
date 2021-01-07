% WTT: what's this for?
clc
clear all
close all
Period='RA\'
%%
for NO=3:14
    All_mean_LW=[];
    All_std_LW=[];
    All_mean_RD=[];
    All_std_RD=[];
    All_mean_RA=[];
    All_std_RA=[];
    for j=1:90
        Y=[];
        for mmmm=1:10
            foldpath=['E:\Knee Exoskeleton\��֤LW��RA\MataData\',Period];
            cd(foldpath);
            load(['Feature_',num2str(NO),'_',num2str(mmmm),'.mat']);
            % plot(Data(15:end,end),'r')
            % hold on
            % plot(Label,'b--')
            heel_strike=heel_strike-14;
            heel_off=heel_off-14;
            %%
            Time=1:1:length(Feature);
            for i=2:length(heel_strike)-1
                x=Time(heel_strike(i):heel_strike(i+1)-1);
                y=Feature(heel_strike(i):heel_strike(i+1)-1,j);
                x_1=linspace(x(1),x(end),100);
                y_1=interp1(x,y,x_1,'spline');
                Y=[Y;y_1];
            end
        end
        mean_y_1=mean(Y,1);
        std_y_1=std(Y);
        
        if  Period=='LW\'
            All_mean_LW=[All_mean_LW;mean_y_1];
            All_std_LW=[All_std_LW;std_y_1];
        end
        if  Period=='RD\'
            All_mean_RD=[All_mean_RD;mean_y_1];
            All_std_RD=[All_std_RD;std_y_1];
        end
        if  Period=='RA\'
            All_mean_RA=[All_mean_RA;mean_y_1];
            All_std_RA=[All_std_RA;std_y_1];
        end
    end
    %%
    cd(foldpath);
    if  Period=='LW\'
        save(['Feature_analysis_',num2str(NO),'.mat'],'All_mean_LW','All_std_LW')
    end
    if  Period=='RD\'
        save(['Feature_analysis_',num2str(NO),'.mat'],'All_mean_RD','All_std_RD')
    end
    if  Period=='RA\'
        save(['Feature_analysis_',num2str(NO),'.mat'],'All_mean_RA','All_std_RA')
    end
end
%%
% xx=1:1:100;
% cd('E:\Knee Exoskeleton\stand_walk_ramp_stair\transparentmode\ѵ��ʱ���������л�\ģ��ѵ��\�����˶�\Program\Left')
%%
% for i=1:18
%     subplot(6,3,i)
%     shade_plot1(xx,All_mean_LW(i*5-0,:),All_std_LW(i*5-0,:),[0.5 0.5 0.5])
% %     hold on
% %     shade_plot2(xx,All_mean_RD(i*5-0,:),All_std_RD(i*5-0,:),[0.8 0.8 0.8])
% end
%%
% load('Feature_analysis.mat');
% ch=4
% for i=1:18
%     subplot(6,3,i)
%     plot(xx,All_mean_RD(i*5-ch,:),'B')
% end
%%





