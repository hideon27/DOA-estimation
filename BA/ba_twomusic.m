%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPEA115
% Project Title: Implementation of Standard Bees Algorithm in MATLAB
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



%% Bees Algorithm Parameters

MaxIt=1000;          % Maximum Number of Iterations

nScoutBee=30;                           % Number of Scout Bees

nSelectedSite=round(0.5*nScoutBee);     % Number of Selected Sites

nEliteSite=round(0.4*nSelectedSite);    % Number of Selected Elite Sites

nSelectedSiteBee=round(0.5*nScoutBee);  % Number of Recruited Bees for Selected Sites

nEliteSiteBee=2*nSelectedSiteBee;       % Number of Recruited Bees for Elite Sites

r=0.1*(VarMax-VarMin);	% Neighborhood Radius

rdamp=0.95;             % Neighborhood Radius Damp Rate



%% 搜索过程
angel = [];                     % 搜索获得的角度
count=0;counta=1;               % count表示得到的局部最优解的个数，counta表示找到的最优个数
BestC = [];                     % 存储找到的所有最优功率和所有最优角度
BestP = [];


while counta <= iwave                   % 信源数即为局部最优解的个数
    r=0.1*(VarMax-VarMin);	% 每次迭代更新
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

    % Empty Bee Structure
    empty_bee.Position=[];
    empty_bee.Cost=[];

    % Initialize Bees Array
    bee=repmat(empty_bee,nScoutBee,1);

    % Create New Solutions
    for i=1:nScoutBee
        bee(i).Position=unifrnd(VarMin,VarMax,VarSize);

        phim1=rad*bee(i).Position(1);
        phim2=rad*bee(i).Position(2);
        a1=exp(-1i*twpi*d.'*sin(phim1)*cos(phim2));
        a2=exp(-1i*twpi*d1.'*sin(phim1)*sin(phim2));
        a=[a1;a2];
        SP=1/(a'*(Un*Un')*a);
        SP=abs(SP);
        bee(i).Cost=-SP;                    %计算评价值

    end

    % Sort
    [~, SortOrder]=sort([bee.Cost]);
    bee=bee(SortOrder);

    % Update Best Solution Ever Found
    BestSol=bee(1);                        %cost值越小越优

    % Array to Hold Best Cost Values
    BestCost=[];
    BestPosition=[];

    %% Bees Algorithm Main Loop

    for it=1:MaxIt

        % Elite Sites
        for i=1:nEliteSite                      %对前i个最优站点进行更新

            bestnewbee.Cost=inf;                %为所有精英蜂群找到的最优解

            for j=1:nEliteSiteBee
                newbee.Position=PerformBeeDance(bee(i).Position,r);

                phim1=rad*newbee.Position(1);
                phim2=rad*newbee.Position(2);
                a1=exp(-1i*twpi*d.'*sin(phim1)*cos(phim2));
                a2=exp(-1i*twpi*d1.'*sin(phim1)*sin(phim2));
                a=[a1;a2];
                SP=1/(a'*(Un*Un')*a);
                SP=abs(SP);
                newbee.Cost=-SP;                    %计算评价值


                if newbee.Cost<bestnewbee.Cost
                    bestnewbee=newbee;
                end
            end

            if bestnewbee.Cost<bee(i).Cost     %以此来对当前站点最优解进行更新
                bee(i)=bestnewbee;
            end

        end

        % Selected Non-Elite Sites
        for i=nEliteSite+1:nSelectedSite       %对非精英站点的处理

            bestnewbee.Cost=inf;

            for j=1:nSelectedSiteBee
                newbee.Position=PerformBeeDance(bee(i).Position,r);

                phim1=rad*newbee.Position(1);
                phim2=rad*newbee.Position(2);
                a1=exp(-1i*twpi*d.'*sin(phim1)*cos(phim2));
                a2=exp(-1i*twpi*d1.'*sin(phim1)*sin(phim2));
                a=[a1;a2];
                SP=1/(a'*(Un*Un')*a);
                SP=abs(SP);
                newbee.Cost=-SP;                    %计算评价值

                if newbee.Cost<bestnewbee.Cost
                    bestnewbee=newbee;
                end
            end

            if bestnewbee.Cost<bee(i).Cost
                bee(i)=bestnewbee;
            end

        end

        % Non-Selected Sites
        for i=nSelectedSite+1:nScoutBee
            bee(i).Position=unifrnd(VarMin,VarMax,VarSize);

            phim1=rad*bee(i).Position(1);
            phim2=rad*bee(i).Position(2);
            a1=exp(-1i*twpi*d.'*sin(phim1)*cos(phim2));
            a2=exp(-1i*twpi*d1.'*sin(phim1)*sin(phim2));
            a=[a1;a2];
            SP=1/(a'*(Un*Un')*a);
            SP=abs(SP);
            bee(i).Cost=-SP;                    %计算评价值


        end

        % Sort
        [~, SortOrder]=sort([bee.Cost]);
        bee=bee(SortOrder);

        % Update Best Solution Ever Found
        BestSol=bee(1);

        % Store Best Cost Ever Found

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


        % Damp Neighborhood Radius
        r=r*rdamp;

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