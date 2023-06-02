%% SA 模拟退火
tic
clear; clc
%% DOA参数
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


%% 搜索过程
angel = [];                     % 搜索获得的角度
count=0;counta=1;               % count表示得到的局部最优解的个数，counta表示找到的最优个数
BestC = [];                     % 存储找到的所有最优功率和所有最优角度
BestP = [];
while counta <= iwave                   % 信源数即为局部最优解的个数
    count=count+1;
    disp(['--->第',num2str(count),'次尝试寻找最优角度']);

    %% 参数初始化
    narvs = 2; % 变量个数
    T0 = 100;   % 初始温度
    T = T0; % 迭代中温度会发生改变，第一次迭代时温度就是T0
    T_Limited=1e-15;  %迭代温度的下限
    maxgen = 1000;  % 最大迭代次数
    Lk = 100;  % 每个温度下的迭代次数
    alfa = 0.95;  % 温度衰减系数
    x_lb = [0 0]; % x的下界
    x_ub = [90 90]; % x的上界

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


    
    %%  随机生成一个初始解

    x0 = zeros(1,narvs);
    for i = 1: narvs
        x0(i) = x_lb(i) + (x_ub(i)-x_lb(i))*rand(1);       %随机初始值是一个在上下界之间的数
    end

    %% Evaluation
    phim1=rad*x0(1);
    phim2=rad*x0(2);
    a1=exp(-1i*twpi*d.'*sin(phim1)*cos(phim2));
    a2=exp(-1i*twpi*d1.'*sin(phim1)*sin(phim2));
    a=[a1;a2];
    SP=1/(a'*(Un*Un')*a);
    %SP=abs(SP);
    y0=SP;          %得到当前角度位置的功率值



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


            phim1=rad*x1(1);
            phim2=rad*x1(2);
            a1=exp(-1i*twpi*d.'*sin(phim1)*cos(phim2));
            a2=exp(-1i*twpi*d1.'*sin(phim1)*sin(phim2));
            a=[a1;a2];
            SP=1/(a'*(Un*Un')*a);
            %SP=abs(SP);
            y1=SP;     % 计算新解的函数值


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
                ' Angle1 = ' num2str(MAXX(it,1)) ' Angle2 = ' num2str(MAXX(it,2)) ]);

            % 判断是否连续10个都相差在1度范围内,是则跳出，
            % 判断是否陷入局部，超过10次角度接近相似，跳出循环，重新初始化寻找
            if all(abs(MAXX(it-9:it-1,:)-MAXX(it,:))<=1)
                break;
            end
        end

    end


    %% 判断寻找到的角度是否重复，如果重复则不存储
    if count == 1
        angel(counta,:) = MAXX(iter,:);
    else
        if all(abs(angel-MAXX(iter,:))>1)   %判断是否找到相似角度，超过1度，不相似可以存储
            angel(counta,:) = MAXX(iter,:);
        else
            continue;
        end
    end
    disp([' * 找到', num2str(counta),' 组最优角度']);

    %% 计算功率
    %MAXY
    %abs(MAXY)

    CostRate=abs(MAXY);
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
    BestP=[BestP;MAXX];

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