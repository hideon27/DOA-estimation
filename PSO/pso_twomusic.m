%%DOA估计--pso算法：L型阵列问题
clc;clear;close all;
cd(fileparts(mfilename('fullpath')));
addpath(genpath(cd));

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

%% PSO Parameters
MaxIt=100;       % Maximum Number of Iterations
nPop=50;        % Population Size (Swarm Size)

% PSO Parameters
w=1;            % Inertia Weight
wdamp=0.99;     % Inertia Weight Damping Ratio
c1=1.5;         % Personal Learning Coefficient
c2=1.5;         % Global Learning Coefficient

% If you would like to use Constriction Coefficients for PSO,
% uncomment the following block and comment the above set of parameters.

% % Constriction Coefficients
% phi1=2.05;
% phi2=2.05;
% phi=phi1+phi2;
% chi=2/(phi-2+sqrt(phi^2-4*phi));
% w=chi;          % Inertia Weight
% wdamp=1;        % Inertia Weight Damping Ratio
% c1=chi*phi1;    % Personal Learning Coefficient
% c2=chi*phi2;    % Global Learning Coefficient

% Velocity Limits
VelMax=0.5*(VarMax-VarMin);
VelMin=-VelMax;

%% 搜索过程
angel = [];                     % 搜索获得的角度
count=0;counta=1;               % count表示第count次寻找得到的局部最优解的个数，counta表示找到的最优个数
BestC = [];                     % 存储找到的所有最优功率和所有最优角度   
BestP = [];
while counta <= iwave                   % 信源数即为局部最优解的个数
    w=1;                                              %每次循环重置w参数值
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
    empty_particle.Position=[];
    empty_particle.Cost=[];
    empty_particle.Velocity=[];
    empty_particle.Best.Position=[];
    empty_particle.Best.Cost=[];

    particle=repmat(empty_particle,nPop,1);

    GlobalBest.Cost=inf;
    GlobalBest.Position =zeros(1,nVar);
    for i=1:nPop

        % Initialize Position
        particle(i).Position =unifrnd(VarMin,VarMax,VarSize);
        % Initialize Velocity
        particle(i).Velocity = zeros(VarSize);
        
        % Evaluation
        phim1=rad*particle(i).Position(1);
        phim2=rad*particle(i).Position(2);
        a1=exp(-1i*twpi*d.'*sin(phim1)*cos(phim2));
        a2=exp(-1i*twpi*d1.'*sin(phim1)*sin(phim2));
        a=[a1;a2];
        SP=1/(a'*(Un*Un')*a);
        SP=abs(SP);
        particle(i).Cost=-SP;

        % Update Personal Best
        particle(i).Best.Position=particle(i).Position;
        particle(i).Best.Cost=particle(i).Cost;

        % Update Global Best
        if particle(i).Best.Cost<GlobalBest.Cost
            GlobalBest=particle(i).Best;                 %将结构体直接赋值
        end
    end

    BestCost=[];
    BestPosition=[];
    %% PSO Main Loop
    for it=1:MaxIt
        for i=1:nPop
            % Update Velocity
            particle(i).Velocity = w*particle(i).Velocity ...
                +c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
                +c2*rand(VarSize).*(GlobalBest.Position-particle(i).Position);

            % Apply Velocity Limits
            particle(i).Velocity = max(particle(i).Velocity,VelMin);
            particle(i).Velocity = min(particle(i).Velocity,VelMax);

            % Update Position
            particle(i).Position = particle(i).Position + particle(i).Velocity;

            % Velocity Mirror Effect
            IsOutside=(particle(i).Position<VarMin | particle(i).Position>VarMax);
            particle(i).Velocity(IsOutside)=-particle(i).Velocity(IsOutside);

            % Apply Position Limits
            particle(i).Position = max(particle(i).Position,VarMin);
            particle(i).Position = min(particle(i).Position,VarMax);

            % Evaluation
            phim1=rad*particle(i).Position(1);
            phim2=rad*particle(i).Position(2);
            a1=exp(-1i*twpi*d.'*sin(phim1)*cos(phim2));
            a2=exp(-1i*twpi*d1.'*sin(phim1)*sin(phim2));
            a=[a1;a2];
            SP=1/(a'*(Un*Un')*a);
            SP=abs(SP);
            particle(i).Cost=-SP;

            % Update Personal Best
            if particle(i).Cost<particle(i).Best.Cost

                particle(i).Best.Position=particle(i).Position;
                particle(i).Best.Cost=particle(i).Cost;

                % Update Global Best
                if particle(i).Best.Cost<GlobalBest.Cost
                    GlobalBest=particle(i).Best;
                end
            end
        end

        BestCost(it,:) = GlobalBest.Cost;
        BestPosition(it,:) = GlobalBest.Position;
        
        if mod(it,5) == 0  % 每隔5次数打印一次
            disp(['  Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it)) ...
                ' Angle1 = ' num2str(BestPosition(it,1)) ' Angle2 = ' num2str(BestPosition(it,2)) ]);
            
            % 判断是否连续5个都相差在0.25度范围内,是则跳出，
            % 判断是否陷入局部，超过5次角度接近相似，跳出循环，重新初始化寻找
            if all(abs(BestPosition(it-4:it-1,:)-BestPosition(it,:))<=1)                    
                break;
            end
        end

        w=w*wdamp;
    end   

    %% 判断寻找到的角度是否重复，如果重复则不存储
    if count == 1
        angel(counta,:) = GlobalBest.Position;  
    else        
        if all(abs(angel-GlobalBest.Position)>1)   %判断是否找到相似角度，超过1.5度，不相似可以存储
            angel(counta,:) = GlobalBest.Position;  
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
    title(['第',num2str(counta),'次找到最优',' angle1: ',num2str(GlobalBest.Position(1)),' angle2: ',num2str(GlobalBest.Position(2))]);
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