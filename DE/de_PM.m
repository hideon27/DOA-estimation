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
nVar=1;             % Number of Decision Variables
VarSize=[1 nVar];   % Size of Decision Variables Matrix
VarMin=-90;         % Lower Bound of Variables
VarMax= 90;         % Upper Bound of Variables
derad=pi/180;                             %角度->弧度
redeg=180/pi;                             %弧度->角度
twpi=2*pi;
kelm=16;                                   %阵元数
dd=0.5;                                   %阵元间距
d=0:dd:(kelm-1)*dd;
iwave=4;                                  %信源数
theta=[-60  0  30  65];                   %波达方向测试数据
pw=[1 0.8 0.7 0.6]';                      %信号功率,维度需要保持和theta一样
nv=ones(1,kelm);                          %归一化噪声方差
snr=20;                                   %信噪比
snr0=10^(snr/10);
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
    A=exp(-1i*twpi*d.'*sin(theta*derad));     %方向向量
    K=length(d);                              %k其实等于阵元数
    cr=zeros(K,K);
    L=length(theta);                          %L为测试角度的总个数
    data=randn(L,n);
    data=sign(data);
    s=diag(pw)*data;                          %信源信号3*200                   
    received_signal=A*s;                      %接受信号
    cx=received_signal+diag(sqrt(nv/snr0/2))*(randn(K,n)+1i*randn(K,n));%增加噪声
    Rxx=cx*cx'/n;
    G=Rxx(:,1:iwave);
    H=Rxx(:,iwave+1:end);
    P=inv(G'*G)*G'*H;                         %传播算子矩阵
    Q=[P',-diag(ones(1,kelm-iwave))];         %Q矩阵

    %% Initialization

    empty_individual.Position=[];
    empty_individual.Cost=[];

    BestSol.Cost=inf;

    pop=repmat(empty_individual,nPop,1);

    for i=1:nPop

        pop(i).Position=unifrnd(VarMin,VarMax,VarSize);

        phim=derad*pop(i).Position;
        a=exp(-1i*twpi*d*sin(phim)).';
        SP=1/(a'*(Q'*Q)*a);
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

            phim=derad*NewSol.Position;
            a=exp(-1i*twpi*d*sin(phim)).';
            SP=1/(a'*(Q'*Q)*a);
            SP=abs(SP);
            NewSol.Cost=-SP;                      %突变之后的评估函数


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
                ' Angle = ' num2str(BestPosition(it))]);

            % 判断是否连续5个都相差在1度范围内,是则跳出，
            % 判断是否陷入局部，超过5次角度接近相似，跳出循环，重新初始化寻找
            if all(abs(BestPosition(it-4:it-1)-BestPosition(it))<=1)
                break;
            end
        end

    end

    %% 判断寻找到的角度是否重复，如果重复则不存储
    if count == 1
        angel(counta) = BestPosition(it);         
    else        
        if all(abs(angel-BestPosition(it))>1)   %判断是否找到相似角度，超过1度，不相似可以存储
            angel(counta) = BestPosition(it);
        else
            continue; 
        end
    end
    disp([' * 找到', num2str(counta),' 个最优角度']);

    % 计算功率
    CostRate=abs(BestCost);
    CostRate_Max=max(CostRate);
    CostRate=10*log10(CostRate/CostRate_Max);

    % 绘画出适应度函数值
    figure;
    plot(CostRate,'LineWidth',2);
    title(['第',num2str(counta),'次找到最优',' angle tends to ',num2str(angel(counta))]);
    xlabel('Iteration');
    ylabel('magnitude(dB)');

    % 存储找到的最优功率和最优角度
    BestC=[BestC;CostRate];
    BestP=[BestP;BestPosition];

    counta = counta+1;

end

figure; % 真实与实验获得对比
plot(sort(angel),'r*');
hold on
plot(sort(theta),'b.');
title(' 真实与实验获得对比 ');
xlabel('Numbers');
ylabel('angels');
set(gca,'XTick',1:iwave)
grid on;

figure; % 所有寻找到的功率变化
plot(BestC,'LineWidth',2);
title('所有寻找到的功率变化');
xlabel('Iteration');
ylabel('magnitude/(dB)');

figure; % 所有寻找到的角度变化
plot(BestP,'LineWidth',2);
title('所有寻找到的角度变化');
xlabel('Iteration');
ylabel('angle/(度)');
angel