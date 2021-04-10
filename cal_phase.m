function cal_phase=cal_phase(mea_CSI)
    len=length(mea_CSI);
    temp(:)=mea_CSI(1,1,:);
    mea_phase=angle(temp);
    Y=zeros(1,len); Y(1)=mea_phase(1);
    cycle=0;
    for i=2:len
        if(mea_phase(i)>mea_phase(i-1)+pi)cycle=cycle+1;end
        Y(i)=-cycle*2*pi+mea_phase(i);
    end
    X=[-28:2:-2 -1 1 2:2:28];
    k=(mean(X.*Y)-mean(X).*mean(Y))./(mean(X.*X)-mean(X).*mean(X));
    b=mean(Y)-k*mean(X);
    cal_phase=Y-k*X-b;
%     figure(1);
%     plot(X,mea_phase);
%     figure(2);
%     plot(X,Y,X,k*X+b,X,cal_phase);
end