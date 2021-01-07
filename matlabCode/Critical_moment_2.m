clear all
close all
%% All the parameters
%##########################################################################
% change it when processing different LMs
LM_FOLDER = 'LW/';
%##########################################################################
BASE_FOLDER = '/Users/tiantong/Desktop/LMR/ICL_LMR/processedData/';

%##########################################################################
% if the number of files in a folder changes, edit it!
nFilePerFolder = 10;
%##########################################################################

%%
%##########################################################################
% iFolder must be edited according to different LMs
for iFolder = 1
%##########################################################################
    
    for jFile = 1: nFilePerFolder

        foldPath=[BASE_FOLDER, LM_FOLDER];
        cd(foldPath);
        load(['converted_',num2str(iFolder),'_',num2str(jFile),'.mat']);
        
        % sample rate = 100 hz
        fs=100;   
        % cutoff freq of lowpass filter = 10 hz
        lowpass_freq=10;
        % normalized cutoff freq
        Wn=lowpass_freq/(fs/2);
        % order of the filter
        n=2;
        [b,a]=butter(n,Wn,'low');
        % WTT: only filter heel/toe force? Do we have to filter IMU data? 
        % Answer: No need to if they are directly feeded into the model
        heel_force_filt=filtfilt(b, a,heel_force);
        toe_force_filt=filtfilt(b, a,toe_force);
        
        % WTT: How are the thresholds (4, 0.5) determined, do they change along
        % with different terrains?
        % Answer: heuristic
        heel_left_temp=heel_force_filt;
        heel_left_temp(heel_left_temp<=4)=0;
        heel_left_temp(heel_left_temp>4)=1;
        
        toe_left_temp=toe_force_filt;
        toe_left_temp(toe_left_temp<=0.5)=0;
        toe_left_temp(toe_left_temp>0.5)=1;
        % %
        heel_strike=[];heel_off=[];toe_strike=[];toe_off=[];
        for i=1:length(heel_left_temp)-1
            if heel_left_temp(i)==0 &&heel_left_temp(i+1)==1
                heel_strike=[heel_strike;i+1];
            end
            if heel_left_temp(i)==1 &&heel_left_temp(i+1)==0
                heel_off=[heel_off;i+1];
            end
        end
        
        for i=2:length(toe_left_temp)-1
            % WTT: why does toe strike/off different from heel strike/off?
            if toe_left_temp(i-1)==0&&toe_left_temp(i)==0 &&toe_left_temp(i+1)==1
                toe_strike=[toe_strike;i+1];
            end
            if toe_left_temp(i-1)==1 &&toe_left_temp(i)==1 &&toe_left_temp(i+1)==0
                toe_off=[toe_off;i+1];
            end
        end
        
        Fasle_1=diff(toe_off);
        [a_1 b_1]=sort(Fasle_1);
        
        Fasle_2=diff(toe_strike);
        [a_2 b_2]=sort(Fasle_2);
        
        Fasle_3=diff(heel_off);
        [a_3 b_3]=sort(Fasle_3);
        
        Fasle_4=diff(heel_strike);
        [a_4 b_4]=sort(Fasle_4);
        %%
        % WTT: how are the thresholds determined?
        % WTT: ???
        
        Fasle_temp_1=find(Fasle_1<44);
        toe_off(Fasle_temp_1+1)=[];
        Fasle_temp_2=find(Fasle_2<65);
        toe_strike(Fasle_temp_2)=[];
        
        Fasle_temp_3=find(Fasle_3<30);
        heel_off(Fasle_temp_3+1)=[];
        Fasle_temp_4=find(Fasle_4<30);
        heel_strike(Fasle_temp_4)=[];
        % %     ��֤ heel_strike(1)<toe_strike(1)<heel_off(1)<toe_off(1)
        
%         % WTT: processing the very beginning point?
%         Position_1=find(toe_strike>heel_strike(1));
%         if length(Position_1)==length(toe_strike)
%         else
%             toe_strike(1:Position_1(1)-1)=[];
%         end
%         
%         Position_2=find(heel_off>toe_strike(1));
%         if length(Position_2)==length(heel_off)
%         else
%             heel_off(1:Position_2(1)-1)=[];
%         end
%         
%         Position_3=find(toe_off>heel_off(1));
%         if length(Position_3)==length(toe_off)
%         else
%             toe_off(1:Position_3(1)-1)=[];
%         end
        
%         if heel_strike(2)<toe_off(1)
%             heel_strike(1)=[];
%             toe_strike(1)=[];
%             heel_off(1)=[];
%         end
        
        figure(iFolder)
        subplot(nFilePerFolder,1,jFile)
        plot(heel_force_filt,'r')
        hold on
        plot(heel_strike,heel_force_filt(heel_strike),'g.')
        hold on
        plot(heel_off,heel_force_filt(heel_off),'k.')
%         subplot(2,1,2)
%         plot(toe_force_filt,'r')
%         hold on
%         plot(toe_strike,toe_force_filt(toe_strike),'b.')
%         hold on
%         plot(toe_off,toe_force_filt(toe_off),'k.')
        
        cd(foldPath);
        save(['Data_',num2str(iFolder),'_',num2str(jFile),'.mat'],'heel_force_filt','toe_force_filt','toe_strike','toe_off','toe_force','heel_force','IMU_Shank','IMU_Thigh','heel_strike','heel_off','Knee_angle');
        save(['Data_',num2str(iFolder),'_',num2str(jFile),'.mat'],'heel_force_filt','IMU_Shank','IMU_Thigh','heel_strike','Knee_angle');
    end
end