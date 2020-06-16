%fixed number of source samples=50, Neural network model
rng('default')
np=50; %np stands for number of source samples
nq_train=600; %nq_train stands for number of target training samples
%nq_crossval=100;
nq_test=200; %number of test target points
mu=.0001; %step size of gradient descent
k=30;

lambda=1;
number_epo=1000; %number of iterations of gradient descent
d=200;
sigma_noise=1; %noise level
sigma_source=2*eye(d); %covariance matrix of source
sigma_target=eye(d); %covariance matrix of target
M_S = normrnd(0,10,[k,d]); % M_S is the source parameter

A=eye([k,1]); %Hidden-to-output layer


M_T2=M_S+normrnd(0,.001,[k,d]); %M_T2 is the target_1 parameter

M_T5=M_S+60000*normrnd(0,.0001,[k,d]); %M_T5 is the target_2 parameter

n2=norm(M_S-M_T2,2);

n5=norm(M_S-M_T5,2);
trial=10;%number of trials
error_target2_trial=zeros([1,601]);
error_target5_trial=zeros([1,601]);
for trial=1:10

Ze=zeros(1,d);
Ze1=0;
%Generating target and source data
X_source=mvnrnd(Ze,sigma_source,np);
y_source=(A.'*poslin(M_S*X_source.')+mvnrnd(Ze1,sigma_noise*eye(1),np).').';

X_train_target2=mvnrnd(Ze,sigma_target,nq_train);
y_train_target2=(A.'*poslin(M_T2*X_train_target2.')+mvnrnd(Ze1,sigma_noise*eye(1),nq_train).').';

X_train_target5=mvnrnd(Ze,sigma_target,nq_train);
y_train_target5=(A.'*poslin(M_T5*X_train_target5.')+mvnrnd(Ze1,sigma_noise*eye(1),nq_train).').';

X_test_target2=mvnrnd(Ze,sigma_target,nq_test);
y_test_target2=(A.'*poslin(M_T2*X_test_target2.')+mvnrnd(Ze1,sigma_noise*eye(1),nq_test).').';

X_test_target5=mvnrnd(Ze,sigma_target,nq_test);
y_test_target5=(A.'*poslin(M_T5*X_test_target5.')+mvnrnd(Ze1,sigma_noise*eye(1),nq_test).').';



M_T_init= normrnd(0,1,[k,d]);
M_T2_est=M_T_init;

M_T5_est=M_T_init;


train_loss_target2=zeros([1,600+1]);
test_loss_target2=zeros([1,600+1]);
train_loss_target5=zeros([1,600+1]);
test_loss_target5=zeros([1,600+1]);
t=zeros(1,np+1);

for i=1:600+1
   t(1,i)=i-1;
end


for number_q=0:600 %sweeping number of target samples
    M_T2_est=M_T_init;
    M_T5_est=M_T_init;
    %number_p=1;
for epoch=1:number_epo %loop for gradeint descent
    
    
    
    
    if number_q>0
       
    r1= A.'*poslin(M_T2_est*X_train_target2(1:number_q,:).')-y_train_target2(1:number_q,:).';
    r2=A.'*poslin(M_T2_est*X_source.')-y_source.';
   
    g2=(1/number_q)*diag(A)*(1/2)*(1+sign(M_T2_est*X_train_target2(1:number_q,:).'))*diag(r1)*X_train_target2(1:number_q,:)+(1/np)*diag(A)*(1/2)*(1+sign(M_T2_est*X_source.'))*diag(r2)*X_source;
    end
    
    if number_q==0
        r2=A.'*poslin(M_T2_est*X_source.')-y_source.';
        g2=(1/np)*diag(A)*(1/2)*(1+sign(M_T2_est*X_source.'))*diag(r2)*X_source;
    end
        
    M_T2_est=M_T2_est-(number_q+1)*mu*g2;
    
    
    if number_q>0
   
    r1= A.'*poslin(M_T5_est*X_train_target5(1:number_q,:).')-y_train_target5(1:number_q,:).';
    r2=A.'*poslin(M_T5_est*X_source.')-y_source.';
    g5=(1/number_q)*diag(A)*(1/2)*(1+sign(M_T5_est*X_train_target5(1:number_q,:).'))*diag(r1)*X_train_target5(1:number_q,:)+(0.001/np)*diag(A)*(1/2)*(1+sign(M_T5_est*X_source.'))*diag(r2)*X_source;
    end
    
    if number_q==0
     
       r2=A.'*poslin(M_T5_est*X_source.')-y_source.';
       g5=(1/np)*diag(A)*(1/2)*(1+sign(M_T5_est*X_source.'))*diag(r2)*X_source;
    end
        
    M_T5_est=M_T5_est-(number_q+1)*mu*g5;
    
 
end
%test error:
    for i=1:nq_test
        test_loss_target2(1,number_q+1)=test_loss_target2(1,number_q+1)+(1/nq_test)*norm(y_test_target2(i,:).'-A.'*poslin(M_T2_est*X_test_target2(i,:).'),2)^2;
    end
    
  %test error: 
    for i=1:nq_test
        test_loss_target5(1,number_q+1)=test_loss_target5(1,number_q+1)+(1/nq_test)*norm(y_test_target5(i,:).'-A.'*poslin(M_T5_est*X_test_target5(i,:).'),2)^2;
    end
   
end


error_target2_trial(1,:)=error_target2_trial(1,:)+test_loss_target2(1,:);
error_target5_trial(1,:)=error_target5_trial(1,:)+test_loss_target5(1,:);


end
figure(1)
plot(t,error_target2_trial/10)
hold on
plot(t,error_target5_trial/10)
legend("Small Delta","Large Delta")
title("fixed np")
xlabel("n_q")
ylabel("generalization error")

