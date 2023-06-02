%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPEA107
% Project Title: Implementation of Differential Evolution (DE) in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
%
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
%
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

clc;
clear;
close all;

%% Problem Definition
nVar=2;             % Number of Decision Variables,两个变量第一变量是theta，第二变量是fe
VarSize=[1 nVar];   % Size of Decision Variables Matrix
VarMin= 0;          % Lower Bound of Variables,注意上下限要保持一致
VarMax= 90;         % Upper Bound of Variables
rad=pi/180;                               %角度->弧度
deg=180/pi;                               %弧度->角度
twpi=2*pi;
kelm=8;                                   %阵元数
dd=0.5;                                   %阵元间距
d=0:dd:(kelm-1)*dd;                       %X轴阵元分布
d1=dd:dd:(kelm-1)*dd;                     %Y轴阵元分布

iwave = 5;                                   %信源数
theta =[10 30 50 60 80];                     %待观测真实角度数据值1
fe=[15 25 5 50 30];                          %待观测真实角度数据值2

snr=10;                                   %信噪比
n=200;                                    %采样数

%% DE Parameters

MaxIt=1000;      % Maximum Number of Iterations

nPop=50;        % Population Size

beta_min=0.2;   % Lower Bound of Scaling Factor
beta_max=0.8;   % Upper Bound of Scaling Factor

pCR=0.2;        % Crossover Probability


%% 搜索过程
angel = [];                     % 搜索获得的角度
count=0;counta=1;               % count表示得到的局部最优解的个数，counta表示找到的最优个数
BestC = [];                     % 存储找到的所有最优功率和所有最优角度
BestP = [];
while counta <= iwave                   % 信源数即为局部最优解的个数
    count=count+1;
    disp(['--->第',num2str(count),'次尝试寻找最优角度']);

    %% Function And Init
    Ax=exp(-1i*twpi*d.'*(sin(theta*rad).*cos(fe*rad)));%X轴上阵元对应的方向矩阵
    Ay=exp(-1i*twpi*d1.'*(sin(theta*rad).*sin(fe*rad)));%Y轴上阵元对应的方向矩阵
    A=[Ax;Ay];
    S=randn(iwave,n);
    X=A*S;                                            %接受信号
    X1=awgn(X,snr,'measured');                        %加入高斯白噪声
    Rxx=X1*X1'/n;                                     %自相关函数
    [EV,D]=eig(Rxx);                                  %求矩阵的特征向量和特征值
    [EVA,I]=sort(diag(D).');                          %特征值按升值排序
    EV=fliplr(EV(:,I));                               %左右翻转，特征值按降序排序
    Un=EV(:,iwave+1:end);                             %噪声子空间

    %% Initialization

    empty_individual.Position=[];
    empty_individual.Cost=[];

    BestSol.Cost=inf;

    pop=repmat(empty_individual,nPop,1);

    for i=1:nPop

        pop(i).Position=unifrnd(VarMin,VarMax,VarSize);


        phim1=rad*pop(i).Position(1);
        phim2=rad*pop(i).Position(2);
        a1=exp(-1i*twpi*d.'*sin(phim1)*cos(phim2));
        a2=exp(-1i*twpi*d1.'*sin(phim1)*sin(phim2));
        a=[a1;a2];
        SP=1/(a'*(Un*Un')*a);
        SP=abs(SP);
        pop(i).Cost=-SP;                    %计算评价值

        if pop(i).Cost<BestSol.Cost
            BestSol=pop(i);                   %评估值越小越好
        end

    end

    BestCost=[];
    BestPosition=[];

    %% DE Main Loop

    for it=1:MaxIt

        for i=1:nPop

            x=pop(i).Position;

            AA=randperm(nPop);

            AA(AA==i)=[];

            a=AA(1);
            b=AA(2);
            c=AA(3);

            % Mutation the ist item excepted
            %beta=unifrnd(beta_min,beta_max);
            beta=unifrnd(beta_min,beta_max,VarSize);
            y=pop(a).Position+beta.*(pop(b).Position-pop(c).Position);

            %apply limitation
            y = max(y, VarMin);
            y = min(y, VarMax);

            % Crossover
            z=zeros(size(x));
            j0=randi([1 numel(x)]);           %从位置变量中任选出一个索引
            for j=1:numel(x)
                if j==j0 || rand<=pCR
                    z(j)=y(j);                %y为变异之后的值
                else
                    z(j)=x(j);                %x为原来的值
                end
            end

            NewSol.Position=z;

            phim1=rad*NewSol.Position(1);
            phim2=rad*NewSol.Position(2);
            a1=exp(-1i*twpi*d.'*sin(phim1)*cos(phim2));
            a2=exp(-1i*twpi*d1.'*sin(phim1)*sin(phim2));
            a=[a1;a2];
            SP=1/(a'*(Un*Un')*a);
            SP=abs(SP);
            NewSol.Cost=-SP;                     %突变之后的评估函数


            if NewSol.Cost<pop(i).Cost
                pop(i)=NewSol;

                if pop(i).Cost<BestSol.Cost
                    BestSol=pop(i);              %更新全局最优解，最小越好
                end
            end

        end

        % Update Best Cost

        BestCost(it,:) = BestSol.Cost;
        BestPosition(it,:) = BestSol.Position;


        if mod(it,5) == 0  % 每隔5次数打印一次
  
            disp(['  Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it)) ...
                ' Angle1 = ' num2str(BestPosition(it,1)) ' Angle2 = ' num2str(BestPosition(it,2)) ]);
            
            % 判断是否连续5个都相差在0.25度范围内,是则跳出，
            % 判断是否陷入局部，超过5次角度接近相似，跳出循环，重新初始化寻找
            if all(abs(BestPosition(it-4:it-1,:)-BestPosition(it,:))<=1)                    
                break;
            end
        end

    end

    %% 判断寻找到的角度是否重复，如果重复则不存储
    if count == 1
        angel(counta,:) = BestPosition(it,:);         
    else        
        if all(abs(angel-BestPosition(it,:))>1)   %判断是否找到相似角度，超过1度，不相似可以存储
            angel(counta,:) = BestPosition(it,:);
        else
            continue; 
        end
    end
    disp([' * 找到 ', num2str(counta),' 组最优角度']);

    % 计算功率
    CostRate=abs(BestCost);
    CostRate_Max=max(CostRate);
    CostRate=10*log10(CostRate/CostRate_Max);

    % 绘画出适应度函数值
    figure;
    plot(CostRate,'LineWidth',2);
    title(['第',num2str(counta),'次找到最优',' angle1: ',num2str(angel(counta,1)),' angle2: ',num2str(angel(counta,2))]);
    xlabel('Iteration');
    ylabel('magnitude(dB)');

    % 存储找到的最优功率和最优角度
    BestC=[BestC;CostRate];
    BestP=[BestP;BestPosition];
    
    counta = counta+1;        
end

figure; % 真实与实验获得对比
[sorted_theta_fe,order1]=sortrows([theta',fe']);
[sorted_angel,order2]=sortrows(angel);   %以第一个角的大小进行排序
             

plot(sorted_theta_fe(:,1),sorted_theta_fe(:,2),"r",'LineWidth',2);
hold on

plot(sorted_theta_fe(:,1),sorted_theta_fe(:,2),'*b','MarkerSize',10,'LineWidth',1.5)
hold on

scatter(sorted_angel(:,1),sorted_angel(:,2),'g','LineWidth',1.5);

legend('真实角度波形','真实角度','实验角度')
title(' 真实与实验获得对比 ');
xlabel('theta');
ylabel('fe');
set(gca,'XTick',0:15:90)
set(gca,'YTick',0:15:90)
grid on;

figure; % 所有寻找到的功率变化
plot(BestC,'LineWidth',2);
title('所有寻找到的功率变化');
xlabel('Iteration');
ylabel('magnitude/(dB)');

figure; % 所有寻找到的角度变化
plot(BestP(:,1),'LineWidth',2);
hold on
plot(BestP(:,2),'LineWidth',2);
legend('theta','fe')
title('所有寻找到的角度变化');
xlabel('Iteration');
ylabel('angle/(度)');
angel