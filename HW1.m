clear all;
close all;
clc;

q=zeros(2000,10);%�W�ҵ����A1~10���������������v��������

t=zeros(2000,10);%�C�x�����ӧO�Q�襤����(�ĴXruns,�����s��)  greedy method
t1=zeros(2000,10);%epsilon-greedy method,epsilon=0.1
t2=zeros(2000,10);%epsilon-greedy method,epsilon=0.01
t3=zeros(2000,10);% UCB

Q=zeros(1,10);%�C�x�����ثe�έp�����������(1�C,�����s��)  greedy method
Q1=zeros(1,10);%epsilon-greedy method,epsilon=0.1
Q2=zeros(1,10);%epsilon-greedy method,epsilon=0.01
Q3=zeros(1,10);%UCB

R=zeros(2000,1000);%reward�έp(�ĴXruns,�ĴX���Ա�)  greedy method
R1=zeros(2000,1000);%epsilon-greedy method,epsilon=0.1
R2=zeros(2000,1000);%epsilon-greedy method,epsilon=0.01
R3=zeros(2000,1000);%UCB

tt=zeros(2000,1000);%��ܥ��T(�̰�reward)���Ա��ܴX�� greedy method
tt1=zeros(2000,1000);%epsilon-greedy method,epsilon=0.1
tt2=zeros(2000,1000);%epsilon-greedy method,epsilon=0.01


for h=1:2000%�@2000runs
    for i=1:10
        q(h,i)=normrnd(0,1);%��C�x������@�Ӱ�����������
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  �Ĥ@����
    k=randi(10);%�Ĥ@�����H����
    
    %greedy method
    t(h,k)=t(h,k)+1;%���x��������+1
    R(h,1)=normrnd(q(h,k),1);%��Ureward
    Q(h,k)=R(h,1);%�Ĥ@���ԩҥHreward=�����
    
    %epsilon-greedy method,epsilon=0.1
    t1(h,k)=t1(h,k)+1;%���x��������+1
    R1(h,1)=normrnd(q(h,k),1);%��Ureward
    Q1(h,k)=R1(h,1);%�Ĥ@���ԩҥHreward=�����
    
    %epsilon-greedy method,epsilon=0.01
    t2(h,k)=t2(h,k)+1;
    R2(h,1)=normrnd(q(h,k),1);
    Q2(h,k)=R2(h,1);
    
    %UCB
    t3(h,k)=t3(h,k)+1;
    R3(h,1)=normrnd(q(h,k),1);
    Q3(h,k)=R3(h,1);
    
    
        [nn lom]=max(q(h,:));%lom=�W�ҵ����̤j�����Ȫ����x�O�ĴX�x�A�p�G��ܨ쨺�x����tt�N+1�A�N����쥿�T���A�]���n�έp��ܥ��T�v
        if k==lom
            tt(h,1)=1;
            tt1(h,1)=1;
            tt2(h,1)=1;
        end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  greedy method
        for j=2:1000%time=2����ϥ�greedy method��̤j�����Q�����x��
            [nn lom]=max(q(h,:));%�W�ҵ����ݳ̤j�����(q*)������
            [n w]=max(Q(h,:));%�ثe�έp��w�x�O�̤j�����
            t(h,w)=t(h,w)+1;%�Բ�w�x�����ƥ[�@
            R(h,j)=normrnd(q(h,w),1);%��U��reward
            Q(h,w)=Q(h,w)+(R(h,j)-Q(h,w))/t(h,w);%�p�⥭��reward(�����) 
            
            if  w==lom
                tt(h,j)=1;%����time2����Ԩ쥿�T�ɶ�����
            end  
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   epsilon-greedy method  0.1    
            
            [n1 w1]=max(Q1(h,:));%�̤j�O���x?
            random=rand();
            kk=randi(10);
            if random<=0.9
                   ww=w1; %��̤ܳj����Ȩ��x
                else 
                    ww=kk;%�H�����
            end
            t1(h,ww)=t1(h,ww)+1;%�Ӿ�������+1
            R1(h,j)=normrnd(q(h,ww),1);%��reward
            Q1(h,ww)=Q1(h,ww)+(R1(h,j)-Q1(h,ww))/t1(h,ww);%�έp�ثe�����
            

            if  ww==lom
                tt1(h,j)=1;%��쪺��n�O�̤j����Ȩ��x���ܸӯx�}����=1
            end

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   epsilon-greedy method 0.01        
            [n2 w2]=max(Q2(h,:));%�̤j�O���x?
            random2=rand();
            kkk=randi(10);
            if random2<=0.99
                   www=w2; %��̤ܳj�έp����Ȩ��x
                else 
                    www=kkk;%�H�����
            end
            t2(h,www)=t2(h,www)+1;%�Ӿ�������+1
            R2(h,j)=normrnd(q(h,www),1);%��reward
            Q2(h,www)=Q2(h,www)+(R2(h,j)-Q2(h,www))/t2(h,www);%�έp�ثe�����
            
            
            if  www==lom
                tt2(h,j)=1;%��쪺��n�O�̤j����Ȩ��x���ܸӯx�}����=1
            end
            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ucb
            [n3 w3]=max(Q3(h,:));%�ثe�̤j�έp����ȬO���x?
            random3=rand();
            kkkk=randi(10);
            for i=1:10
                test(h,i)=Q3(h,i)+(2*sqrt(log(j)/t3(h,i)));%UCB����
                
                    if t3(h,i)==0%�����U�Ь�0��X�ӭȷ|�L���j�A�ҥH�ڱN�U�Ь�0������X���L���j�A���]��10
                        test(h,i)=10;
                    end
                    
                [nnnnn wucb]=max(test(h,:)); %wucb��UCB�⧹��x�}���ȳ̤j���Ӧ�m   
            end
            
            if random3<=0.9%epsilon-greedy method  0.1
                   wwww=w3; %
                else 
                    wwww=wucb;%�����ɿ��wucb
            end
            t3(h,wwww)=t3(h,wwww)+1;
            R3(h,j)=normrnd(q(h,wwww),1);
            Q3(h,wwww)=Q3(h,wwww)+(R3(h,j)-Q3(h,wwww))/t3(h,wwww);%�έp�ثe�����
        
        end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  �e��

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

