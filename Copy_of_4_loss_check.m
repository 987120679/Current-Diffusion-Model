% This script checks the training and validation loss in order to prevent
% overfitting and help to choose the optimal model
% This script file should be put under the pycharm project with root './checkpoints/'
% e.g., './checkpoints/loss_check.m'

% Date:2020/2/28

clear
clc
close all
% MNIST_1*ssim+5*phy+lr0.0002
% /home/zhaoh/PycharmProjects/BP_inv_physic_inspired (som)/checkpoints/MNIST_1*ssim+5*phy+lr0.001+1net/loss_log_clean.txt
model='demo';%the project name to be checked
%% extract training and validation losses
fileroot=['./results/' model '/'];
filename='log1014.txt';%the file saves training losses
% parentPath = ['./results/' model '/val_TriSpace16_diff_syn_L31_40/val_metric.mat'];
fid = fopen([fileroot filename]);
data = textscan(fid,'%s %s %s %s %s %s %s %s %f', 'Delimiter',' ');
train_loss = string(data{6});
% load(parentPath)
% train_loss=loss_log_clean(:,end-4:end);%the last two columns correspond to training loss;(loss_phy,loss_ima,loss_12)
% val_loss=metric(:,end-1:end);%the last two columns correspond to validation loss: [ssim rmse],[psnr,ssim rmse]
% val_loss=metric(:,:,:);% correspond to validation loss: [psnr,ssim rmse]
%% visualize
iter=length(train_loss);%number of epoches
iter=100000;
% iter=size(train_loss,1);%number of total iterations
train_loss_data = ones([iter,1]);
for i=1:iter
    temp = train_loss(i);
    temp_data = erase(temp,",");
    train_loss_data(i) = str2double(temp);
end
train_loss_smooth=smooth(train_loss_data, ceil(iter/300),'moving');
%%
h=figure;
%yyaxis left
semilogy(linspace(1,iter,iter),train_loss_data,'color',[0.93 0.93 0.93],'HandleVisibility','off')
hold on
semilogy(linspace(1,iter,iter),train_loss_smooth,'k','LineStyle','-')
xlabel('Epoch number'),ylabel('Loss')
legend('training loss')
title('Training loss')
saveas(gcf,'loss_.fig')
saveas(h,'loss_.jpg')
hold off

%%
clear
clc
close all

%% extract training and validation losses
model='MDLS_block5';%the model name to be checked
fileroot=['./checkpoints/' model '/'];
filename='loss_log_clean.txt';%the file saves training losses
parentPath = ['./results/' model '/val_30/val_metric.mat'];
load([fileroot filename]);
load(parentPath)
train_loss_MDLS=loss_log_clean(:,end-4:end);%the last two columns correspond to training loss;(loss_phy,loss_ima,loss_12)
% val_loss=metric(:,end-1:end);%the last two columns correspond to validation loss: [ssim rmse],[psnr,ssim rmse]
val_loss_MDLS=metric(:,:,:);% correspond to validation loss: [psnr,ssim rmse]

model='ICLM_bpdouble';%the model name to be checked
fileroot=['./checkpoints/' model '/'];
filename='loss_log_clean.txt';%the file saves training losses
parentPath = ['./results/' model '/val_30/val_metric.mat'];
load([fileroot filename]);
load(parentPath)
train_loss_Somnet=loss_log_clean(:,end-4:end);%the last two columns correspond to training loss;(loss_phy,loss_ima,loss_12)
% val_loss=metric(:,end-1:end);%the last two columns correspond to validation loss: [ssim rmse],[psnr,ssim rmse]
val_loss_Somnet=metric(:,:,:);% correspond to validation loss: [psnr,ssim rmse]
%% visualize
epoch=size(metric,1);%number of epoches
iter=size(train_loss_MDLS,1);%number of total iterations
for k=1:5
    train_loss_MDLS_smooth=smooth(train_loss_MDLS(:,k),ceil(iter/500),'moving');
    train_loss_Somnet_smooth=smooth(train_loss_Somnet(:,k),ceil(iter/500),'moving');
    if k==2
        val_loss_MDLS(:,k)=1-val_loss_MDLS(:,k); %1-ssim to be consistent with training loss
        val_loss_Somnet(:,k)=1-val_loss_Somnet(:,k);
    elseif k==4
        val_loss_MDLS(:,k)=val_loss_MDLS(:,k-1); 
        val_loss_Somnet(:,k)=val_loss_Somnet(:,k-1); 
    elseif k==5
         val_loss_MDLS(:,k)=val_loss_MDLS(:,k-2);
         val_loss_Somnet(:,k)=val_loss_Somnet(:,k-1); 
    end
    h=figure;
    yyaxis left
    semilogy(linspace(1,epoch,iter),train_loss_Somnet_smooth,'k')
    hold on
    semilogy(linspace(1,epoch,iter),train_loss_MDLS_smooth,'r')
    xlabel('Epoch number'),ylabel('Loss')
    yyaxis right
    hold on
    plot(1:epoch,val_loss_MDLS(:,k),'r','linewidth',1.0)
    hold on
    plot(1:epoch,val_loss_Somnet(:,k),'g','linewidth',1.0)
    if k==1
        legend('training L_{E} loss','validation:PSNR')
%         legend('training L_1 loss','validation:1-SSIM')
    elseif k==2
        legend('training L_{ssim} loss','validation:1-SSIM')
    elseif k==3
        legend('training L_{mse} loss','validation:RMSE')
    elseif k==4
        legend('training L_{J} loss','validation:RMSE')
    else    
        legend('training L_{phy+ssim+mse+J} loss','validation:RMSE')
    end
    title(['Training loss(' num2str(k) ')' ' vs Validation loss('  num2str(k) ')'])
    saveas(gcf,['loss_'  num2str(k) '.fig'])
    saveas(h,['loss_'  num2str(k) '.jpg'])
    hold off
end

