%% 适用于多行数据
function test_scale_my= data_scale_myself(test_data,ymin,ymax,xmax,xmin)
[mtest,ntest] = size(test_data);
dataset=test_data;
dataset=dataset';
for i=1:size(dataset,1)
%     xmin(i,1)=min(dataset(i,:));
%     xmax(i,1)=max(dataset(i,:));
    dataset_scale(i,:)=(ymax-ymin)*(dataset(i,:)-xmin(i,1))/(xmax(i,1)-xmin(i,1)) + ymin;
end
dataset_scale=dataset_scale';
test_scale_my = dataset_scale(1:mtest,:);



















