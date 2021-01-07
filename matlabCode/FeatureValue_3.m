clc
clear all
close all
%% All the parameters
LM_FOLDER = 'RA/';
BASE_FOLDER = '/Users/tiantong/Desktop/LMR/ICL_LMR/processedData/';
foldPath=[BASE_FOLDER, LM_FOLDER];
cd(foldPath);

for NO=3:14
    for mmmm=1:10
        load(['Data_',num2str(NO),'_',num2str(mmmm),'.mat'])
        Data=[IMU_Thigh IMU_Shank];
        
        
        Cnt_Data=size(Data, 1); 
        Cnt_Buffer=15;
        Cnt_Inc=1;
        Cnt_Channel=1;
        Cnt_Feature=5;
        % WTT: Data_Buffer size of 15 * 1
        Data_Buffer=zeros(Cnt_Buffer, Cnt_Channel);
        Cnt_Features=length(Cnt_Buffer:Cnt_Inc:Cnt_Data);
        Feature=zeros(Cnt_Features,Cnt_Channel*Cnt_Feature);
        k=0;
        %%
        % WTT: didn't remove yaw channel? 
        for i=Cnt_Buffer:Cnt_Inc:Cnt_Data  
            % WTT: Data_Buffer size of 15 * 18
            Data_Buffer=Data(i-Cnt_Buffer+1:i,1:size(Data,2));
            k=k+1;
            for j=1:size(Data_Buffer,2)
                Feature(k,(5*j-4))=max(Data_Buffer(:,j));
                Feature(k,(5*j-3))=min(Data_Buffer(:,j));
                Feature(k,(5*j-2))=mean(Data_Buffer(:,j));
                Feature(k,(5*j-1))=std(Data_Buffer(:,j));
                Feature(k,(5*j))=sqrt(mean(Data_Buffer(:,j).^2));
            end
        end
        if  Period=='LW/'
            group=1;
        end
        if  Period=='RD/'
            group=2;
        end
        if  Period=='RA/'
            group=3;
        end
        Label=group*ones(length(Feature),1);
        cd(foldpath);
        % WTT: what's the point?
        Data=Data(Cnt_Buffer:end,:);
        save(['Feature_',num2str(NO),'_',num2str(mmmm),'.mat'],'heel_strike','heel_off','Data','Label','Feature');
    end
end