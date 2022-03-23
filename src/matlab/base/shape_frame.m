function shaped_frame=shape_frame(x,sequence_length,fs)

energy=sum(abs(x));
x=x./energy;

shaped_frame=zeros(1,size(x,2)*3,size(x,1));

I=real(x);
Q=imag(x);
psd=zeros(size(I));

I = permute(I,[2 1]);
Q = permute(Q,[2 1]);

nt=size(x,2);

shaped_frame(1,1:nt,:)=I;
shaped_frame(1,nt+1:2*nt,:)=Q;

for i=1:size(psd,2)
    [S,F,T,P]=spectrogram(x(:,i),hanning(sequence_length), ...
             0,sequence_length,fs,'centered');
    psd(:,i)=10*log10(abs(fftshift(P)));

end
     
psd=permute(psd,[2 1]);
   
shaped_frame(1,2*nt+1:end,:)=psd;

end