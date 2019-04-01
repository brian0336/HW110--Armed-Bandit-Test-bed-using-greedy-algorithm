clear all;
close all;
clc;

q=zeros(2000,10);%上帝視角，1~10號機器的中獎機率的平均值

t=zeros(2000,10);%每台機器個別被選中次數(第幾runs,機器編號)  greedy method
t1=zeros(2000,10);%epsilon-greedy method,epsilon=0.1
t2=zeros(2000,10);%epsilon-greedy method,epsilon=0.01
t3=zeros(2000,10);% UCB

Q=zeros(1,10);%每台機器目前統計的中獎期望值(1列,機器編號)  greedy method
Q1=zeros(1,10);%epsilon-greedy method,epsilon=0.1
Q2=zeros(1,10);%epsilon-greedy method,epsilon=0.01
Q3=zeros(1,10);%UCB

R=zeros(2000,1000);%reward統計(第幾runs,第幾次拉桿)  greedy method
R1=zeros(2000,1000);%epsilon-greedy method,epsilon=0.1
R2=zeros(2000,1000);%epsilon-greedy method,epsilon=0.01
R3=zeros(2000,1000);%UCB

tt=zeros(2000,1000);%選擇正確(最高reward)的拉桿選擇幾次 greedy method
tt1=zeros(2000,1000);%epsilon-greedy method,epsilon=0.1
tt2=zeros(2000,1000);%epsilon-greedy method,epsilon=0.01


for h=1:2000%共2000runs
    for i=1:10
        q(h,i)=normrnd(0,1);%把每台機器抽一個高斯分布的值
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  第一次拉
    k=randi(10);%第一次抽隨機拉
    
    %greedy method
    t(h,k)=t(h,k)+1;%那台機器次數+1
    R(h,1)=normrnd(q(h,k),1);%當下reward
    Q(h,k)=R(h,1);%第一次拉所以reward=期望值
    
    %epsilon-greedy method,epsilon=0.1
    t1(h,k)=t1(h,k)+1;%那台機器次數+1
    R1(h,1)=normrnd(q(h,k),1);%當下reward
    Q1(h,k)=R1(h,1);%第一次拉所以reward=期望值
    
    %epsilon-greedy method,epsilon=0.01
    t2(h,k)=t2(h,k)+1;
    R2(h,1)=normrnd(q(h,k),1);
    Q2(h,k)=R2(h,1);
    
    %UCB
    t3(h,k)=t3(h,k)+1;
    R3(h,1)=normrnd(q(h,k),1);
    Q3(h,k)=R3(h,1);
    
    
        [nn lom]=max(q(h,:));%lom=上帝視角最大中獎值的那台是第幾台，如果選擇到那台機器tt就+1，代表有選到正確的，因為要統計選擇正確率
        if k==lom
            tt(h,1)=1;
            tt1(h,1)=1;
            tt2(h,1)=1;
        end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  greedy method
        for j=2:1000%time=2之後使用greedy method選最大期望值Q的那台拉
            [nn lom]=max(q(h,:));%上帝視角看最大期望值(q*)的機器
            [n w]=max(Q(h,:));%目前統計第w台是最大期望值
            t(h,w)=t(h,w)+1;%拉第w台的次數加一
            R(h,j)=normrnd(q(h,w),1);%當下的reward
            Q(h,w)=Q(h,w)+(R(h,j)-Q(h,w))/t(h,w);%計算平均reward(期望值) 
            
            if  w==lom
                tt(h,j)=1;%紀錄time2之後拉到正確時間次數
            end  
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   epsilon-greedy method  0.1    
            
            [n1 w1]=max(Q1(h,:));%最大是哪台?
            random=rand();
            kk=randi(10);
            if random<=0.9
                   ww=w1; %選擇最大期望值那台
                else 
                    ww=kk;%隨機選擇
            end
            t1(h,ww)=t1(h,ww)+1;%該機器次數+1
            R1(h,j)=normrnd(q(h,ww),1);%抽reward
            Q1(h,ww)=Q1(h,ww)+(R1(h,j)-Q1(h,ww))/t1(h,ww);%統計目前期望值
            

            if  ww==lom
                tt1(h,j)=1;%選到的剛好是最大期望值那台的話該矩陣元素=1
            end

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   epsilon-greedy method 0.01        
            [n2 w2]=max(Q2(h,:));%最大是哪台?
            random2=rand();
            kkk=randi(10);
            if random2<=0.99
                   www=w2; %選擇最大統計期望值那台
                else 
                    www=kkk;%隨機選擇
            end
            t2(h,www)=t2(h,www)+1;%該機器次數+1
            R2(h,j)=normrnd(q(h,www),1);%抽reward
            Q2(h,www)=Q2(h,www)+(R2(h,j)-Q2(h,www))/t2(h,www);%統計目前期望值
            
            
            if  www==lom
                tt2(h,j)=1;%選到的剛好是最大期望值那台的話該矩陣元素=1
            end
            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ucb
            [n3 w3]=max(Q3(h,:));%目前最大統計期望值是哪台?
            random3=rand();
            kkkk=randi(10);
            for i=1:10
                test(h,i)=Q3(h,i)+(2*sqrt(log(j)/t3(h,i)));%UCB公式
                
                    if t3(h,i)==0%公式下標為0算出來值會無限大，所以我將下標為0的項算出的無限大，都設成10
                        test(h,i)=10;
                    end
                    
                [nnnnn wucb]=max(test(h,:)); %wucb為UCB算完後矩陣內值最大那個位置   
            end
            
            if random3<=0.9%epsilon-greedy method  0.1
                   wwww=w3; %
                else 
                    wwww=wucb;%探索時選擇wucb
            end
            t3(h,wwww)=t3(h,wwww)+1;
            R3(h,j)=normrnd(q(h,wwww),1);
            Q3(h,wwww)=Q3(h,wwww)+(R3(h,j)-Q3(h,wwww))/t3(h,wwww);%統計目前期望值
        
        end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  畫圖

f=mean(R);
f1=mean(R1);
f2=mean(R2);
x = [1:1:1000];
figure(1);
plot(x,f,'g',x,f1,'b',x,f2,'r');
legend('greedy method','epsilon greedy method=0.1','epsilon greedy method=0.01');
hold on;

ttt=sum(tt)/2000;
ttt1=sum(tt1)/2000;
ttt2=sum(tt2)/2000;
x = [1:1:1000];
figure(2);
set(gca,'yticklabel',{'0%','10%', '20%', '30%', '40%' ,'50%','60%','70%','80%','90%','100%'});
plot(x,ttt,'g',x,ttt1,'b',x,ttt2,'r');
legend('greedy method','epsilon greedy method=0.1','epsilon greedy method=0.01');
hold on;

fucb=mean(R3);
x = [1:1:1000];
figure(3);
plot(x,fucb,'b',x,f1,'c');
legend('ucb,c=2','epsilon greedy method=0.1');
hold on;

