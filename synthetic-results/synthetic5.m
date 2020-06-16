%fixed number of source samples=300 and number of target samples=20, 
%linear model
rng('default')
np=300; %np stands for number of source samples
nq_train=20; %nq_train stands for number of target training samples

nq_test=200; %number of test target points
nq_val=50; %nq_val stands for number of target validation set
mu=.038; %step size of gradient descent
k=30;
lambda=1;
number_epo=6500; %number of iterations of gradient descent
d=50;
Ze=zeros(1,d);
Ze1=zeros(1,k);
sigma_noise=.3; %noise level
sigma_source=2*eye(d);  %covariance matrix of source
sigma_target=eye(d); %covariance matrix of target

M_T5 = normrnd(0,10,[k,d]); %M_T5 is the target parameter
MM=normrnd(0,.0001,[k,d]); % source parameter=target parameter + c* MM
ff=zeros([350,2]);
ll=0;
%Generating target data
X_train_target5=mvnrnd(Ze,sigma_target,nq_train);
y_train_target5=(M_T5*X_train_target5.'+mvnrnd(Ze1,sigma_noise*eye(k),nq_train).').';



X_val_target5=mvnrnd(Ze,sigma_target,nq_val);
y_val_target5=(M_T5*X_val_target5.'+mvnrnd(Ze1,sigma_noise*eye(k),nq_val).').';


X_test_target5=mvnrnd(Ze,sigma_target,nq_test);
y_test_target5=(M_T5*X_test_target5.'+mvnrnd(Ze1,sigma_noise*eye(k),nq_test).').';
for kk=1:400:140000 %sweeping transfer distance
ll=ll+1;


M_S=M_T5+kk*MM; % M_S is the source parameter


EST_M1=zeros([k,d]);
EST_M2=zeros([k,d]);
EST_M3=zeros([k,d]);
EST_M4=zeros([k,d]);
EST_M5=zeros([k,d]);

ff(ll,1)=norm(M_S-M_T5,2);

for trial=1:20



%generating source data
X_source=mvnrnd(Ze,sigma_source,np);
y_source=(M_S*X_source.'+mvnrnd(Ze1,sigma_noise*eye(k),np).').';



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
    

for epoch=1:650  %loop for gradeint descent
    
   
    g5=(2/nq_train)*(M_T5_est*X_train_target5.'*X_train_target5-y_train_target5.'*X_train_target5)+(2*cc/np)*(M_T5_est*X_source.'*X_source-y_source.'*X_source);
    
        
    M_T5_est=M_T5_est-mu*g5;
   
   
end
%testing on validation set
    for i=1:nq_val
        gg(1,yy)=gg(1,yy)+(1/nq_val)*norm(y_val_target5(i,:).'-M_T5_est*X_val_target5(i,:).',2)^2;
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
        ff(ll,2)=ff(ll,2)+(1/nq_test)*norm(y_test_target5(i,:).'-M_T5_est*X_test_target5(i,:).',2)^2;
    end
   
    end

ff(ll,2)=ff(ll,2)/20;

end


bb=sortrows(ff,1); %sorting based on the transfer distance
figure(6)
plot(bb(:,1),bb(:,2))
hold on


title("np=300 and nq=20")
xlabel("Delta")
ylabel("generalization error")

