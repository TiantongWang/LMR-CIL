function [Matrix, Accuracy]=Counfusion_Matrix(Groups, Label_Ref, Label_Predict)

Num_Mode=length(Groups);
Matrix=zeros(Num_Mode,Num_Mode);
for i=1:Num_Mode
    for j=1:Num_Mode
        Matrix(i,j)=sum((Label_Ref==Groups(i)) & (Label_Predict==Groups(j)))/sum((Label_Ref==Groups(i)));
    end
end
% Accuracy=mean(diag(Matrix));
Accuracy=length(find(Label_Predict==Label_Ref))/length(Label_Predict);
