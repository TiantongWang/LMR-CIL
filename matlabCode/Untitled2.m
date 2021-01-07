clear all;close all;clc;
LIE=[1 2 9 10 11 18];
Groups=[1 2 3 4];
allspeed_right_rate=[];
K=10;
for type=1:1:length(Groups)
    if type==1;Period='LW';NO=1;end;
    if type==2;Period='RA';NO=3;end;
    if type==3;Period='RA';NO=4;end;
    if type==4;Period='RA';NO=5;end;
    if type==5;Period='RA';NO=6;end;
    if type==6;Period='RA';NO=7;end;
    if type==7;Period='RA';NO=8;end
    if type==8;Period='RA';NO=9;end;
    if type==9;Period='RA';NO=10;end;
    if type==10;Period='RA';NO=11;end;
    if type==11;Period='RA';NO=12;end;
    if type==12;Period='RA';NO=13;end;
    if type==13;Period='RA';NO=14;end;
    len=0;feature=[];
    
    for j=1:8
        foldpath=['E:\Knee Exoskeleton\验证LW与RA\old_program\MataData\',Period];
        cd(foldpath);load(['All_Feature_',num2str(NO),'.mat'])
        A=cell2mat(All_data{j,1});A=A(:,LIE);feature=[feature;A];
        train_label{type,j,:}=All_label{j,1};len=len+length(cell2mat(All_data{j,1}));
    end;
    
    Number(type)=len;U{type,:}= {mean(feature)'};S{type,:}= {cov(feature)};
    S_1{type,:}= {inv(cell2mat(S{type,:}))};S_d{type,:}= {det(cell2mat(S{type,:}))};
end
%%
for type=1:length(Groups)
    Prior(type)=Number(type)/sum(Number,2);
end
% sum(Prior)
% %
test=[];test_label=[];
for type=1:length(Groups)
    if type==1;Period='LW';NO=1;end;
    if type==2;Period='RA';NO=3;end;
    if type==3;Period='RA';NO=4;end;
    if type==4;Period='RA';NO=5;end;
    if type==5;Period='RA';NO=6;end;
    if type==6;Period='RA';NO=7;end;
    if type==7;Period='RA';NO=8;end
    if type==8;Period='RA';NO=9;end;
    if type==9;Period='RA';NO=10;end;
    if type==10;Period='RA';NO=11;end;
    if type==11;Period='RA';NO=12;end;
    if type==12;Period='RA';NO=13;end;
    if type==13;Period='RA';NO=14;end;
    
    for j=9:K
        foldpath=['E:\Knee Exoskeleton\验证LW与RA\old_program\MataData\',Period];
        cd(foldpath);load(['All_Feature_',num2str(NO),'.mat'])
        A=cell2mat(All_data{j,1});A=A(:,LIE);test=[test;A];
        label=Groups(type)*ones(length(A),1);
        test_label=[test_label;label];
    end;
end
%%
right1=0;error1=0;D=size(LIE,2);
for i=1:size(test,1)
    for type=1:length(Groups)
        ln_P_x_type(i,type)=-D*log(2*pi)/2-(1/2)*log(cell2mat(S_d{type,:}))-...
            1/2*(test(i,:)'-cell2mat(U{type,:}))'*cell2mat(S_1{type,:})*(test(i,:)'-cell2mat(U{type,:}));  %对数似然
        ln_P_x_type_P_type(i,type)=ln_P_x_type(i,type)+log(Prior(type));
    end
    [Pm,ind]=max(ln_P_x_type_P_type(i,:));predict(i,:)=ind;
    if ind==test_label(i); right1=right1+1;else;error1=error1+1;end;
end
right_rate=right1/size(test,1);
cd('E:\Knee Exoskeleton\验证LW与RA\old_program\Program')
[Matrix, Accuracy]=Counfusion_Matrix(Groups, test_label, predict)
allspeed_right_rate=[allspeed_right_rate right_rate];
plot(test_label,'r','linewidth',2)
hold on
plot(predict,'b','linewidth',2)
% axis([0 size(predict,1) 0 14]);
% legend('reference','predict')
% set(gca,'yticklabel',{'','0^o','2.5^o','5^o','7.5^o','10^o',''},'Fontsize',10);
% set(gca,'ytick',0:6,'Fontsize',10);
% set(gca,'ylim',[0 6],'Fontsize',10);
%%
