clc;
clear;
close all;
nprocs_env=3;
figure
% for id=1:nprocs_env
%     learning_curve
% end
id=1;color="b";
learning_curve_;
id=2;color="r";
learning_curve_;
id=3;color="k";
learning_curve_;
legend('',label1,'',label2,'',label3)
xlim([0,1000])