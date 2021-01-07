% WTT: error
clc
clear all
close all
NO_1=1
MMMM=3
%%
for NO=3:14
    NO_2=NO;
    NO_3=NO;
    Terrian_1='LW';
    Terrian_2='RA';
    Terrian_3='RD';
    %%
    subplot(3,4,NO-2)
    for jjj=1:3
        if jjj==1
            Period=Terrian_1;
            foldpath=['/Users/tiantong/Desktop/LMR/IMU_DATA/MataData/',Period];
            cd(foldpath);
            load(['Gaitdata_',num2str(NO_1),'.mat']);
            a_1=mean_toe_strike*ones(length(xx),1);
            a_2=mean_heel_off*ones(length(xx),1);
            a_3=mean_toe_off*ones(length(xx),1);
            cd('/Users/tiantong/Desktop/LMR/IMU_DATA/Program');
            %         shade_plot1(xx,mean_y_1,std_y_1,[0.3 0.3 0.3])
            plot(xx,mean_y_1,'R','Linewidth',2)
            hold on
            plot([mean_toe_strike mean_toe_strike], [min(mean_y_1) max(mean_y_1)],'R-','Linewidth',1);
            hold on
            plot([mean_heel_off mean_heel_off], [min(mean_y_1) max(mean_y_1)],'R--','Linewidth',1);
            hold on
            plot([mean_toe_off mean_toe_off], [min(mean_y_1) max(mean_y_1)],'R.-','Linewidth',1);
            hold on
        end
        if jjj==2
            Period=Terrian_2;
            foldpath=['/Users/tiantong/Desktop/LMR/IMU_DATA/MataData/',Period];
            cd(foldpath);
            load(['Gaitdata_',num2str(NO),'.mat']);
            a_1=mean_toe_strike*ones(length(xx),1);
            a_2=mean_heel_off*ones(length(xx),1);
            a_3=mean_toe_off*ones(length(xx),1);
            cd('/Users/tiantong/Desktop/LMR/IMU_DATA/Program');
            plot(xx,mean_y_1,'B','Linewidth',2)
            hold on
            plot([mean_toe_strike mean_toe_strike], [min(mean_y_1) max(mean_y_1)],'B-','Linewidth',1);
            hold on
            plot([mean_heel_off mean_heel_off], [min(mean_y_1) max(mean_y_1)],'B--','Linewidth',1);
            hold on
            plot([mean_toe_off mean_toe_off], [min(mean_y_1) max(mean_y_1)],'B.-','Linewidth',1);
            hold on
        end
        if jjj==3
            Period=Terrian_3;
            foldpath=['/Users/tiantong/Desktop/LMR/IMU_DATA/MataData/',Period];
            cd(foldpath);
            load(['Gaitdata_',num2str(NO),'.mat']);
            a_1=mean_toe_strike*ones(length(xx),1);
            a_2=mean_heel_off*ones(length(xx),1);
            a_3=mean_toe_off*ones(length(xx),1);
            cd('/Users/tiantong/Desktop/LMR/IMU_DATA/Program');
            plot(xx,mean_y_1,'K','Linewidth',2)
            hold on
            plot([mean_toe_strike mean_toe_strike], [min(mean_y_1) max(mean_y_1)],'K-','Linewidth',1);
            hold on
            plot([mean_heel_off mean_heel_off], [min(mean_y_1) max(mean_y_1)],'K--','Linewidth',1);
            hold on
            plot([mean_toe_off mean_toe_off], [min(mean_y_1) max(mean_y_1)],'K.-','Linewidth',1);
            hold on
        end
    end
%     legend('LW',['RA ',num2str(NO),'��'],['RD ',num2str(NO),'��'])
   title(['red LW, blue RA ',num2str(NO),'�� ', 'black RD ',num2str(NO),'��'])
   xlabel('Gaitcycle (%)')
   ylabel('Kneeangle(^o)')
end
%%

