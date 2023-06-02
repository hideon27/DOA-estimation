%%DOA估计--pso算法
clc;clear;close all;
cd(fileparts(mfilename('fullpath')));
addpath(genpath(cd));
%% Problem Definition
%CostFunction=@(x) Sphere(x);        % Cost Function
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

%% PSO Parameters
MaxIt=100;       % Maximum Number of Iterations
nPop=50;        % Population Size (Swarm Size)

% PSO Parameters
w=1;            % Inertia Weight
wdamp=0.99;     % Inertia Weight Damping Ratio
c1=1.5;         % Personal Learning Coefficient
c2=2.0;         % Global Learning Coefficient

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
count=0;counta=1;               % count表示得到的局部最优解的个数，counta表示找到的最优个数
BestC = [];                     % 存储找到的所有最优功率和所有最优角度   
BestP = [];
while counta <= iwave                   % 信源数即为局部最优解的个数
    w=1;            % Inertia Weight
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
    empty_particle.Position=[];
    empty_particle.Cost=[];
    empty_particle.Velocity=[];
    empty_particle.Best.Position=[];
    empty_particle.Best.Cost=[];

    particle=repmat(empty_particle,nPop,1);

    GlobalBest.Cost=inf;
    GlobalBest.Position=zeros(1,nVar);

    for i=1:nPop

        % Initialize Position
        particle(i).Position=unifrnd(VarMin,VarMax,VarSize);

        % Initialize Velocity
        particle(i).Velocity=zeros(VarSize);

        % Evaluation
        %particle(i).Cost=CostFunction(particle(i).Position);
        phim=derad*particle(i).Position;
        a=exp(-1i*twpi*d*sin(phim)).';
        SP=1/(a'*(Q'*Q)*a);
        SP=abs(SP);
        particle(i).Cost=-SP;

        % Update Personal Best
        particle(i).Best.Position=particle(i).Position;
        particle(i).Best.Cost=particle(i).Cost;

        % Update Global Best
        if particle(i).Best.Cost<GlobalBest.Cost
            GlobalBest=particle(i).Best;
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
            %particle(i).Cost = CostFunction(particle(i).Position);
            phim=derad*particle(i).Position;
            a=exp(-1i*twpi*d*sin(phim)).';
            SP=1/(a'*(Q'*Q)*a);
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
                ' Angle = ' num2str(BestPosition(it))]);
            
            % 判断是否连续5个都相差在1度范围内,是则跳出，
            % 判断是否陷入局部，超过5次角度接近相似，跳出循环，重新初始化寻找
            if all(abs(BestPosition(it-4:it-1)-BestPosition(it))<=1)                    
                break;
            end
        end

        w=w*wdamp;
    end   

    %% 判断寻找到的角度是否重复，如果重复则不存储
    if count == 1
        angel(counta) = GlobalBest.Position;         
    else        
        if all(abs(angel-GlobalBest.Position)>1)   %判断是否找到相似角度，超过1度，不相似可以存储
            angel(counta) = GlobalBest.Position;
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
    title(['第',num2str(counta),'次找到最优',' angle tends to ',num2str(GlobalBest.Position)]);
    xlabel('Iteration');
    ylabel('magnitude(dB)');

    % 存储找到的最优功率和最优角度
    BestC=[BestC;CostRate];
    BestP=[BestP;BestPosition];
    
    counta = counta+1;        
end

figure; % 真实与实验获得对比
plot(sort(theta),"g",'LineWidth',2);
hold on
plot(sort(angel),'r*');
hold on
plot(sort(theta),'b.');
legend('真实角度波形','实验值','真实值');
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