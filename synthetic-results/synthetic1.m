%fixed number of target samples=50, Linear model
rng('default')
np=500; %np stands for number of source samples
nq_train=50; %nq_train stands for number of target training samples
%nq_crossval=100;
nq_test=200; %number of test target points
mu=.00015; %step size of gradient descent
k=30;
lambda=1;
number_epo=1000; %number of iterations of gradient descent
d=200;
sigma_noise=1; %noise level
sigma_source=2*eye(d); %covariance matrix of source
sigma_target=eye(d); %covariance matrix of target
M_S = normrnd(0,10,[k,d]);% M_S is the source parameter

M_T2=M_S+normrnd(0,.001,[k,d]); %M_T2 is the target_1 parameter 

M_T5=M_S+60000*normrnd(0,.0001,[k,d]); %M_T5 is the target_2 parameter

n2=norm(M_S-M_T2,2);

n5=norm(M_S-M_T5,2);
trial=10; %number of trials
error_target2_trial=zeros([1,501]);
error_target5_trial=zeros([1,501]);
for trial=1:10

Ze=zeros(1,d);
Ze1=zeros(1,k);
%Generating target and source data
X_source=mvnrnd(Ze,sigma_source,np);
y_source=(M_S*X_source.'+mvnrnd(Ze1,sigma_noise*eye(k),np).').';

X_train_target2=mvnrnd(Ze,sigma_target,nq_train);
y_train_target2=(M_T2*X_train_target2.'+mvnrnd(Ze1,sigma_noise*eye(k),nq_train).').';

X_train_target5=mvnrnd(Ze,sigma_target,nq_train);
y_train_target5=(M_T5*X_train_target5.'+mvnrnd(Ze1,sigma_noise*eye(k),nq_train).').';

X_test_target2=mvnrnd(Ze,sigma_target,nq_test);
y_test_target2=(M_T2*X_test_target2.'+mvnrnd(Ze1,sigma_noise*eye(k),nq_test).').';

X_test_target5=mvnrnd(Ze,sigma_target,nq_test);
y_test_target5=(M_T5*X_test_target5.'+mvnrnd(Ze1,sigma_noise*eye(k),nq_test).').';



M_T_init= normrnd(0,1,[k,d]);
M_T2_est=M_T_init;

M_T5_est=M_T_init;


train_loss_target2=zeros([1,np+1]);
test_loss_target2=zeros([1,np+1]);
train_loss_target5=zeros([1,np+1]);
test_loss_target5=zeros([1,np+1]);
t=zeros(1,np+1);

for i=1:np+1
   t(1,i)=i-1;
end



for number_p=0:500 %sweeping number of source samples
    M_T2_est=M_T_init;
    M_T5_est=M_T_init;
   
for epoch=1:number_epo %loop for gradeint descent
    
    
    if number_p>0
    g2=(2/nq_train)*(M_T2_est*X_train_target2.'*X_train_target2-y_train_target2.'*X_train_target2)+(2/number_p)*(M_T2_est*X_source(1:number_p,:).'*X_source(1:number_p,:)-y_source(1:number_p,:).'*X_source(1:number_p,:));
    end
    
    if number_p==0
     g2=(2/nq_train)*(M_T2_est*X_train_target2.'*X_train_target2-y_train_target2.'*X_train_target2);
    end
        
    M_T2_est=M_T2_est-(number_p+1)*mu*g2;
    
    if number_p>0
    g5=(2/nq_train)*(M_T5_est*X_train_target5.'*X_train_target5-y_train_target5.'*X_train_target5)+(2/number_p)*(M_T5_est*X_source(1:number_p,:).'*X_source(1:number_p,:)-y_source(1:number_p,:).'*X_source(1:number_p,:));
    end
    
    if number_p==0
     g5=(2/nq_train)*(M_T5_est*X_train_target5.'*X_train_target5-y_train_target5.'*X_train_target5);
    end
        
    M_T5_est=M_T5_est-(number_p+1)*mu*.1*g5;
    
   
  
    
end
%test error:
    for i=1:nq_test
        test_loss_target2(1,number_p+1)=test_loss_target2(1,number_p+1)+(1/nq_test)*norm(y_test_target2(i,:).'-M_T2_est*X_test_target2(i,:).',2)^2;
    end
    test_loss_target2(1,number_p+1)
 %test error   
    for i=1:nq_test
        test_loss_target5(1,number_p+1)=test_loss_target5(1,number_p+1)+(1/nq_test)*norm(y_test_target5(i,:).'-M_T5_est*X_test_target5(i,:).',2)^2;
    end
    test_loss_target5(1,number_p+1)
   

end


error_target2_trial(1,:)=error_target2_trial(1,:)+test_loss_target2(1,:);
error_target5_trial(1,:)=error_target5_trial(1,:)+test_loss_target5(1,:);


end
error_target2_trial(1,:)=error_target2_trial(1,:)/10;
error_target5_trial(1,:)=error_target5_trial(1,:)/10;
figure(1)
plot(t,error_target2_trial)
hold on
plot(t,error_target5_trial)
legend("Small Delta","Large Delta")
title("fixed nq")
xlabel("n_p")
ylabel("generalization error")


