%% SA 模拟退火
tic
clear; clc
%% DOA参数
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

%% 搜索过程
angel = [];                     % 搜索获得的角度
count=0;counta=1;               % count表示得到的局部最优解的个数，counta表示找到的最优个数
BestC = [];                     % 存储找到的所有最优功率和所有最优角度
BestP = [];
while counta <= iwave                   % 信源数即为局部最优解的个数
    count=count+1;
    disp(['--->第',num2str(count),'次尝试寻找最优角度']);

    %% 参数初始化
    narvs = 1; % 变量个数
    T0 = 100;   % 初始温度
    T = T0; % 迭代中温度会发生改变，第一次迭代时温度就是T0
    T_Limited=1e-15;  %迭代温度的下限
    maxgen = 1000;  % 最大迭代次数
    Lk = 100;  % 每个温度下的迭代次数
    alfa = 0.95;  % 温度衰减系数
    x_lb = -90; % x的下界
    x_ub = 90; % x的上界

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


    
    %%  随机生成一个初始解

    x0 = zeros(1,narvs);
    for i = 1: narvs
        x0(i) = x_lb(i) + (x_ub(i)-x_lb(i))*rand(1);       %随机初始值是一个在上下界之间的数
    end

    %% Evaluation
    phim=derad*x0;
    a=exp(-1i*twpi*d*sin(phim)).';
    SP=1/(a'*(Q'*Q)*a);
    %SP=abs(SP);
    y0=SP;     %得到当前角度位置的功率值

    %% 定义一些保存中间过程的量，方便输出结果和画图
    max_y = y0;     % 初始化找到的最佳的解对应的函数值为y0
    best_x = x0;
    MAXY = []; % 记录每一次外层循环结束后找到的max_y (方便画图）
    MAXX = []; % 记录每一次外层循环结束后找到的best_x (方便画图）


    %% 模拟退火过程
    for iter = 1 : maxgen  % 外循环, 我这里采用的是指定最大迭代次数
        for i = 1 : Lk  % 内循环，在每个温度下开始迭代
            y = randn(1,narvs);  % 生成1行narvs列的N(0,1)随机数
            z = y / sqrt(sum(y.^2)); % 根据新解的产生规则计算z
            x_new = x0 + z*T; % 根据新解的产生规则计算x_new的值
            % 如果这个新解的位置超出了定义域，就对其进行调整
            for j = 1: narvs
                if x_new(j) < x_lb(j)
                    r = rand(1);
                    x_new(j) = r*x_lb(j)+(1-r)*x0(j);
                elseif x_new(j) > x_ub(j)
                    r = rand(1);
                    x_new(j) = r*x_ub(j)+(1-r)*x0(j);
                end
            end
            x1 = x_new;    % 将调整后的x_new赋值给新解x1

            phim=derad*x1;
            a=exp(-1i*twpi*d*sin(phim)).';
            SP=1/(a'*(Q'*Q)*a);
            %SP=abs(SP);
            y1 =SP;  % 计算新解的函数值

            if y1 > y0    % 如果新解函数值大于当前解的函数值
                x0 = x1; % 更新当前解为新解
                y0 = y1;
            else
                p = exp(-(y0 - y1)/T); % 根据Metropolis准则计算一个概率
                if rand(1) < p   % 生成一个随机数和这个概率比较，如果该随机数小于这个概率
                    x0 = x1; % 更新当前解为新解
                    y0 = y1;
                end
            end

            % 判断是否要更新找到的最佳的解
            if y0 > max_y  % 如果当前解更好，则对其进行更新
                max_y = y0;  % 更新最大的y
                best_x = x0;  % 更新找到的最好的x
            end
        end


        MAXY(iter,:) = max_y;  % 保存本轮外循环结束后找到的最大的y
        MAXX(iter,:) = best_x; % 保存本轮外循环结束后找到的最大y对应的x位置
        T = alfa*T;          % 温度下降

        it=iter;             % it即为外循环的当前的迭代次数
        if mod(it,10) == 0    % 每隔10次数打印一次
            disp(['  Iteration ' num2str(it) ': Best Cost = ' num2str(MAXY(it)) ...
                ' Angle = ' num2str(MAXX(it))]);

            % 判断是否连续10个都相差在0.05度范围内,是则跳出，
            % 判断是否陷入局部，超过10次角度接近相似，跳出循环，重新初始化寻找
            if all(abs(MAXX(it-9:it-1)-MAXX(it))<=0.05)
                break;
            end
        end

    end


    %% 判断寻找到的角度是否重复，如果重复则不存储
    if count == 1
        angel(counta) = MAXX(iter);
    else
        if all(abs(angel-MAXX(iter))>2)   %判断是否找到相似角度，超过2度，不相似可以存储
            angel(counta) = MAXX(iter);
        else
            continue;
        end
    end
    disp([' * 找到', num2str(counta),' 个最优角度']);

    %% 计算功率
    %MAXY
    %abs(MAXY)
    CostRate=abs(MAXY);

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
    BestP=[BestP;MAXX];

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