close all;
tic

spf = 1024;  % samples per frame
n_signals = 1000;
n_span_sym = 8;
fc_d = 902e6;  % Central frequency for digital modulations
fc_a = 100e6;  % Central frequency for analog modulations
fc_am = 50e3;  % Central frequency for AM
transDelay=50;

applyRandomFrame = false;

% Modulations
modulationTypes = categorical(["BPSK", "QPSK", "8-PSK", ...
    "16-APSK", "32-APSK", "64-APSK", "128-APSK", "256-APSK",...
    "PAM4", "16-QAM", "32-QAM", "64-QAM", "128-QAM", "256-QAM", ... 
    "GFSK", "CPFSK", "OQPSK", "B-FM", "DSB-AM", "SSB-AM"]);

% Number of samples per symbol
%vLs = [2, 4, 8, 16, 16, 32, 32, 32];
%vMs = [1, 1, 1, 1, 5, 1, 3, 5];
vLs = [8];
vMs = [1];
assert(length(vLs) == length(vMs))

% For each signal in mixture, power attenuation is chosen uniform-randomly
% from the given range
% powerRatiosDb = -10:2:10; 

% Clock offset options
applyClockOffset = false;
maxDeltaOff = 5;  % Maximum clock offset of 5 ppm (parts per million)

seed = 0;

for channel = ["AWGN"]%fr, "Rayleigh", "Rician"]
for fs = [0.2e6] %[0.2e6, 0.6e6, 1e6, 1.5e6, 2e6]
for sps_idx = 1:length(vLs)
for RC = [0.35] %[0.15, 0.25, 0.35, 0.45]
    snrs = -6:2:18;
    acc_ml = zeros(length(snrs), length(modulationTypes));
    acc_ml_nf = zeros(length(snrs), length(modulationTypes));
    acc_ml_est = zeros(length(snrs), length(modulationTypes));
for modulation_idx = 1:length(modulationTypes)
for snr_idx = 1:length(snrs)
    snr = snrs(snr_idx);
    rng(seed)
    seed = seed + 1;
    
    Ls = vLs(sps_idx);
    Ms = vMs(sps_idx);
    sps = Ls / Ms;
    modulation = modulationTypes(modulation_idx);

    if ~isfile(sprintf("../../data/signal/%s/%s_%s_%0.1g_%0.1g_%0.2g_%d.mat", "rx_s", channel, ...
            modulation, fs, sps, RC, snr))
        continue
    end
    fprintf("%s - Computing %s_%s_%0.1g_%0.1g_%0.2g_%d.mat \n", ...
        datestr(toc/86400,'HH:MM:SS'), channel, modulation, fs, sps, RC, snr);
    
    loadfun = @(var) load(sprintf("../../data/signal/%s/%s_%s_%0.1g_%0.1g_%0.2g_%d.mat", var, channel, ...
            modulation, fs, sps, RC, snr), var).(var);
    y = loadfun("y");
    yml = loadfun("yml");
    yml_nf = loadfun("yml_nf");
    yml_est = loadfun("yml_est");
    [~, y] = max(y,[],2);
    [~, yml] = max(yml,[],2);
    [~, yml_nf] = max(yml_nf,[],2);
    [~, yml_est] = max(yml_est,[],2);
    acc_ml(snr_idx, modulation_idx) = mean(y == yml);
    acc_ml_nf(snr_idx, modulation_idx) = mean(y == yml_nf);
    acc_ml_est(snr_idx, modulation_idx) = mean(y == yml_est);
    
end
end
end
end
end
end


figure;
hold on

x = snrs;
for i = 1:length(modulationTypes)
y1 = acc_ml(:, i);
plot(x,y1)
end
title('Combine Plots')

hold off


figure;
hold on

x = snrs;
for i = 1:length(modulationTypes)
y1 = acc_ml_nf(:, i);
plot(x,y1)
end
title('Combine Plots')

hold off

figure;
hold on


x = snrs;
[~, idxs] = sort(acc_ml_est(1,:), 'descend');
for i = 1:length(modulationTypes)
y1 = acc_ml_est(:, idxs(i));
plot(x,y1,'DisplayName',string(modulationTypes(idxs(i))))
end
legend
title('Accuracy per class of maximum likelihood')
xlabel('SNR') 
ylabel('Accuracy') 

hold off
