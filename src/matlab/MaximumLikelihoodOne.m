function y_ml = MaximumLikelihoodOne(x, snr, sps)
modulationPool = categorical(["BPSK", "QPSK", "8-PSK", ...
    "16-APSK", "32-APSK", "64-APSK", "128-APSK", "256-APSK",...
    "PAM4", "16-QAM", "32-QAM", "64-QAM", "128-QAM", "256-QAM", ... 
    "GFSK", "CPFSK", "OQPSK", "B-FM", "DSB-AM", "SSB-AM"]);
likelihood = zeros(1,length(modulationPool));
N0 = 10^(-snr/20);
sigma = N0/sqrt(2);
for j = 1:length(modulationPool)
    modulation = modulationPool(j);
    % modulations which I don't know
    % how to compute the maximum likelihood
    if contains(char(modulation), ...
            {'GFSK', 'CPFSK', 'B-FM', 'DSB-AM', 'SSB-AM','OQPSK'})
        likelihood(j) = -inf;
        continue
    end
    M = getM(modulation);
    data = 0:1:getM(modulation)-1;
    % The dummy variables are only used on modulations which I don't know
    % how to compute the maximum likelihood
    mod = getModulator(modulation,-1,-1,-1,-1,-1);
    txSig = mod(data);
    for i = 1:sps:length(x)
        likelihood(j) = likelihood(j) + log10(sum(1/M/(2*pi*sigma^2).*exp(-(abs(x(i)-txSig)).^2/2/(sigma^2))));
    end
end
y_ml = exp(likelihood)/sum(exp(likelihood));

end