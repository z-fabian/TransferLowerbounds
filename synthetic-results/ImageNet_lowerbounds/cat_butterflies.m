%%%% r_S=1.5, sigma^2_min(V)=.048, r_T = 0.14, RANK=1620


yout1=zeros(200,2);%yout_1: crude lower bound
yout2=zeros(200,2);%precise lower bound
upper1=zeros(200,2);
upper1(:,2)=.271;
for i=1:38
    y=5*0.1653^2+.25*.048*13.48^2*.01*[1-.8*.14*i*13.48^2/(0.1653^2*(1620*32-1))];
    y2=5*0.1653^2+.25*.048*(13.48/8)^2*[1-(i*.14*13.48^2/(2*0.1653^2)+log(2))/(1620*32*log(2))];
    %xx=i-50;
    i;
    yout1(i,2)=y;
    yout1(i,1)=i;
    upper1(i,1)=i;
    yout2(i,2)=y2;
    yout2(i,1)=i;
    %yconst(xx,2)=gg2(xx);
    %yconst(xx,1)=i;
    %xx=xx+1;
end

for i=39:200
    y=5*0.1653^2+.25*.048*0.1653^2*(1620*32-1)/(256*.14*i);
    y2=5*0.1653^2+.25*.048*0.1653^2*(1620*32-1)^2*log(2)/(128*i*1620*32*.14);
    %xx=i-50;
    i;
    yout1(i,2)=y;
    yout1(i,1)=i;
    upper1(i,1)=i;
    yout2(i,2)=y2;
    yout2(i,1)=i;
    %yconst(xx,2)=gg2(xx);
    %yconst(xx,1)=i;
    %xx=xx+1;
end


plot(yout1(:,1),yout1(:,2));
title("cat-butterflies")
xlabel("n_T")
ylabel("Generalization Error")
hold on
plot(yout2(:,1),yout2(:,2));
hold on 
plot(upper1(:,1),upper1(:,2),'r--');
dlmwrite('cat-butterflies_lowerbound_crude.dat', yout1,' ');
dlmwrite('cat-butterflies_lowerbound_precise.dat', yout2,' ');
dlmwrite('cat-butterflies_upperbound.dat', upper1,' ');