clc
clear all
close all
NO_1=1
MMMM=3
All_toe_strike_percent=[];
All_heel_off_percent=[];
All_toe_off_percent=[];
%%
for NO=3:14
    NO_2=NO
    NO_3=NO
    Terrian_1='LW';
    Terrian_2='RA';
    Terrian_3='RD';
    toe_strike_percent=[];
    heel_off_percent=[];
    toe_off_percent=[];
    %%
    for jjj=1:3
        if jjj==1
            Period=Terrian_1;
            foldpath=['E:\Knee Exoskeleton\验证LW与RA\MataData\',Period];
            cd(foldpath);
            load(['Gaitdata_',num2str(NO_1),'.mat']);
            toe_strike_percent=[toe_strike_percent mean_toe_strike];
            heel_off_percent=[heel_off_percent mean_heel_off];
            toe_off_percent=[toe_off_percent mean_toe_off];
        end
        if jjj==2
            Period=Terrian_2;
            foldpath=['E:\Knee Exoskeleton\验证LW与RA\MataData\',Period];
            cd(foldpath);
            load(['Gaitdata_',num2str(NO),'.mat']);
            toe_strike_percent=[toe_strike_percent mean_toe_strike];
            heel_off_percent=[heel_off_percent mean_heel_off];
            toe_off_percent=[toe_off_percent mean_toe_off];
        end
        if jjj==3
            Period=Terrian_3;
            foldpath=['E:\Knee Exoskeleton\验证LW与RA\MataData\',Period];
            cd(foldpath);
            load(['Gaitdata_',num2str(NO),'.mat']);
            toe_strike_percent=[toe_strike_percent mean_toe_strike];
            heel_off_percent=[heel_off_percent mean_heel_off];
            toe_off_percent=[toe_off_percent mean_toe_off];
        end
    end
    All_toe_strike_percent=[All_toe_strike_percent;toe_strike_percent];
    All_heel_off_percent=[All_heel_off_percent;heel_off_percent];
    All_toe_off_percent=[All_toe_off_percent;toe_off_percent]; 
end
%%

