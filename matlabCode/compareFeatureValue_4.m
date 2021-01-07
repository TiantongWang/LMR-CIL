clc
clear all
close all
Period='LW\'
foldpath=['E:\Knee Exoskeleton\验证LW与RA\MataData\',Period];
cd(foldpath);
NO=1
load(['Feature_analysis_',num2str(NO),'.mat'])
feature_1=All_mean_LW;
%%
Period='RD\'
foldpath=['E:\Knee Exoskeleton\验证LW与RA\MataData\',Period];
cd(foldpath);
NO=3
load(['Feature_analysis_',num2str(NO),'.mat'])
feature_2=All_mean_RD;
xx=1:1:100;
%%
ch=0
for i=1:18
    subplot(6,3,i)
    plot(xx,feature_1(i*5-ch,:),'r')
    hold on
    plot(xx,feature_2(i*5-ch,:),'b')
end
%%
