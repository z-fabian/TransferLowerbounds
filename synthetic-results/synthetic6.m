%fixed number of source samples=300 and number of target samples=20
%neural network model
rng('default')
np=300; %np stands for number of source samples
nq_train=20; %nq_train stands for number of target training samples

nq_test=200; %number of test target points
nq_val=50; %nq_val stands for number of target validation set
mu=.038; %step size of gradient descent
k=30;
lambda=1;
number_epo=650; %number of iterations of gradient descent
d=50;
sigma_noise=.3; %noise level
sigma_source=2*eye(d); %covariance matrix of source
sigma_target=eye(d); %covariance matrix of target
A=eye([k,1]);%Hidden-to-output layer
Ze=zeros(1,d);
Ze1=0;
M_T5 = normrnd(0,10,[k,d]); %M_T5 is the target parameter
MM=normrnd(0,.0001,[k,d]); % source parameter=target parameter + c* MM
ff=zeros([350,2]);
ll=0;
%Generating target data
X_train_target5=mvnrnd(Ze,sigma_target,nq_train);
y_train_target5=(A.'*poslin(M_T5*X_train_target5.')+mvnrnd(Ze1,sigma_noise*eye(1),nq_train).').';

X_test_target5=mvnrnd(Ze,sigma_target,nq_test);
y_test_target5=(A.'*poslin(M_T5*X_test_target5.')+mvnrnd(Ze1,sigma_noise*eye(1),nq_test).').';


X_val_target5=mvnrnd(Ze,sigma_target,nq_val);
y_val_target5=(A.'*poslin(M_T5*X_val_target5.')+mvnrnd(Ze1,sigma_noise*eye(1),nq_val).').';

for kk=1:400:140000 %sweeping transfer distance
ll=ll+1;
%M_T2=M_S+normrnd(0,.001,[k,d]);

M_S=M_T5+kk*MM; % M_S is the source parameter

%n2=norm(M_S-M_T2,2);
EST_M1=zeros([k,d]);
EST_M2=zeros([k,d]);
EST_M3=zeros([k,d]);
EST_M4=zeros([k,d]);
EST_M5=zeros([k,d]);

ff(ll,1)=norm(M_S-M_T5,2);

for trial=1:20


Ze=zeros(1,d);
Ze1=0;
%generating source data
X_source=mvnrnd(Ze,sigma_source,np);
y_source=(A.'*poslin(M_S*X_source.')+mvnrnd(Ze1,sigma_noise*eye(1),np).').';




M_T_init= zeros([k,d]);


M_T5_est=M_T_init;


train_loss_target2=zeros([1,np+1]);
test_loss_target2=zeros([1,np+1]);
train_loss_target5=zeros([1,np+1]);
test_loss_target5=zeros([1,np+1]);
t=zeros(1,np+1);

for i=1:np+1
   t(1,i)=i-1;
end


    gg=zeros([1,5]);
    for yy=1:5 %tunning the weight of empirical risk using validation data
        if yy==1
            cc=1;
        end
        if yy==2
            cc=.75;
        end
        if yy==3
            cc=.5;
        end
        if yy==4
            cc=.25;
        end
        if yy==5
            cc=0;
        end
        
    M_T5_est=M_T_init;
    

for epoch=1:650 %loop for gradeint descent
    
    
    
    r1= A.'*poslin(M_T5_est*X_train_target5.')-y_train_target5.';
    r2=A.'*poslin(M_T5_est*X_source.')-y_source.';
    g5=(2/nq_train)*diag(A)*(1/2)*(1+sign(M_T5_est*X_train_target5.'))*diag(r1)*X_train_target5+(2*cc/np)*diag(A)*(1/2)*(1+sign(M_T5_est*X_source.'))*diag(r2)*X_source;
    
   
    M_T5_est=M_T5_est-mu*g5;
   
    
    
end
%testing on validation set
    for i=1:nq_val
        gg(1,yy)=gg(1,yy)+(1/nq_val)*norm(y_val_target5(i,:).'-A.'*poslin(M_T5_est*X_val_target5(i,:).'),2)^2;
    end
    if yy==1
    EST_M1=M_T5_est;
    end
    if yy==2
         EST_M2=M_T5_est;
    end
    if yy==3
        EST_M3=M_T5_est;
    end
    if yy==4
        EST_M4=M_T5_est;
    end
    if yy==5
        EST_M5=M_T5_est;
    end
        
        
    

    end
    %finding the optimal weight of empirical risk using validation set
    if gg(1,1)<gg(1,2) && gg(1,1)<gg(1,3) && gg(1,1)<gg(1,4) && gg(1,1)<gg(1,5)
        M_T5_est=EST_M1;
    end
    if gg(1,2)<gg(1,1) && gg(1,2)<gg(1,3) && gg(1,2)<gg(1,4) && gg(1,2)<gg(1,5)
        M_T5_est=EST_M2;
    end
    
    if gg(1,3)<gg(1,1) && gg(1,3)<gg(1,2) && gg(1,3)<gg(1,4) && gg(1,3)<gg(1,5)
        M_T5_est=EST_M3;
    end
    
    if gg(1,4)<gg(1,1) && gg(1,4)<gg(1,2) && gg(1,4)<gg(1,3) && gg(1,4)<gg(1,5)
        M_T5_est=EST_M4;
    end
    if gg(1,5)<gg(1,1) && gg(1,5)<gg(1,2) && gg(1,5)<gg(1,3) && gg(1,5)<gg(1,4)
        M_T5_est=EST_M5;
    end
    
    
    
    
  %test error 
    for i=1:nq_test
        ff(ll,2)=ff(ll,2)+(1/nq_test)*norm(y_test_target5(i,:).'-A.'*poslin(M_T5_est*X_test_target5(i,:).'),2)^2;
    end
  
    end

ff(ll,2)=ff(ll,2)/20;


end


aa=sortrows(ff,1); %sorting based on the transfer distance
figure(1)
plot(aa(:,1),aa(:,2))
hold on


title("np=300 and nq=20")
xlabel("Delta")
ylabel("generalization error")


