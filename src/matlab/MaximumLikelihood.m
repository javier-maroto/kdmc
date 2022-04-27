function y_ml = MaximumLikelihood(x, snr, sps)
modulationPool = categorical(["BPSK", "QPSK", "8-PSK", ...
    "16-APSK", "32-APSK", "64-APSK", "128-APSK", "256-APSK",...
    "PAM4", "16-QAM", "32-QAM", "64-QAM", "128-QAM", "256-QAM", ... 
    "GFSK", "CPFSK", "OQPSK", "B-FM", "DSB-AM", "SSB-AM"]);
likelihood = zeros(size(x,1),length(modulationPool));
N0 = 10.^(-snr/20);
sigma = N0/sqrt(2);

x2 = x(:, 1:sps:size(x,2));
%x2 = x;
for j = 1:length(modulationPool)
    modulation = modulationPool(j);
    % modulations which I don't know
    % how to compute the maximum likelihood
    if contains(char(modulation), ...
            {'GFSK', 'CPFSK', 'B-FM', 'DSB-AM', 'SSB-AM','OQPSK'})
        likelihood(:, j) = -inf;
        continue
    end
    M = getM(modulation);
    data = 0:1:getM(modulation)-1;
    % The dummy variables are only used on modulations which I don't know
    % how to compute the maximum likelihood
    mod = getModulator(modulation,-1,-1,-1,-1,-1);
    txSig = mod(data);
    x3 = [];
    for tx=txSig
        x3 = cat(3, x3, abs(x2-tx));
    end
    sigma2 = sigma.^2;
    likelihood(:, j) = sum(log10(sum(1/M./(2*pi*sigma2).*exp(-x3.^2/2./sigma2),3)),2);
end
y_ml = exp(likelihood)./sum(exp(likelihood),2);

end