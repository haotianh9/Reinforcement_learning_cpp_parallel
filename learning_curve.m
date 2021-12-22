clc;
clear;
% clear a ave_reward_2 r ave_reward window_length
close all;
fidin=fopen('./build/proc2_log.txt'); % 打开test2.txt文件             
fidout=fopen('./build/mkmatlab.txt','w'); % 创建MKMATLAB.txt文件
i=0;
while ~feof(fidin) % 判断是否为文件末尾 
    i=i+1;
	tline=fgetl(fidin); % 从文件读行 
    if i >= 19
        for j=1 : length(tline)
            if (double(tline(j))>=48&&double(tline(j))<=57 || double(tline(j))==46|| double(tline(j))==45) 
                fprintf(fidout,'%s',tline(j)); % 如果是数字行，把此行数据写入文件MKMATLAB.txt
            else
                fprintf(fidout,' '); % 如果是数字行，把此行数据写入文件MKMATLAB.txt
            end
        end
            fprintf(fidout,'\n'); 
    end
end
fclose(fidout);

A=importdata('./build/mkmatlab.txt');
r=A(:,3);

t=length(r);
a=linspace(1,t,t);
scatter(a,r,2.09,'filled','b');

window_length=100;

for i=window_length+1:length(a)
    if isnan(r(i))
        i
        r(i)=0;
    end
    ave_reward(i)=mean(r(i-window_length:i));
    
    
end

for i=1:length(a)-window_length
    if isnan(r(i))
        i
        r(i)=0;
    end
    ave_reward_2(i)=mean(r(i:i+window_length));
    
end

hold on

% plot(a,ave_reward')
plot(a(1:length(a)-window_length),ave_reward_2','b')
% ylim([-10 50])
ax = gca;
ax.FontSize = 19;
xlabel('episodes')
ylabel('reward')
% xlim([0 50000])
mean(r(length(r)-2000:length(r)))