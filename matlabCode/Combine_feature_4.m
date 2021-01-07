% what's this for, and cannot find removement of invalid features
clc
clear all
close all
Period='RA/'
%%
for NO=3:14
    for mmmm=1:10
        foldpath=['/Users/tiantong/Desktop/LMR/IMU_DATA/MataData/',Period];
        cd(foldpath);
        load(['Feature_',num2str(NO),'_',num2str(mmmm),'.mat'])
%             invalid= [1,2,3,5,6,8,17,18,19,20,22,27,29,46,47,48,49,50,51,52,53,55,58,61,62,63,65,67,76,81,82,83];% EXO 90 features stand lw sa sd
%             Feature(:,invalid)=[];
        % WTT: cell
        feature{mmmm,:,:}={Feature};
        label{mmmm,:,:}={Label};
        data{mmmm,:,:}={Data(1:length(Data),:)};
    end
    for mmmm=1:10
        All_feature{mmmm,:,:}=feature{mmmm};
        All_label{mmmm,:,:}=label{mmmm};
        All_data{mmmm,:,:}=data{mmmm};
    end
    save(['All_Feature_',num2str(NO),'.mat'],'All_data','All_feature','All_label')
    clear feature label data All_feature All_label All_data;
end
