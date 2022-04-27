tic
clear all;

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

% Create folders
mkdir('../../data/signal/yml');
mkdir('../../data/signal/yml_nf');
mkdir('../../data/signal/yml_est');


seed = 0;
for channel = ["AWGN"]%fr, "Rayleigh", "Rician"]
for fs = [0.2e6] %[0.2e6, 0.6e6, 1e6, 1.5e6, 2e6]
for sps_idx = 1:length(vLs)
for RC = [0.35] %[0.15, 0.25, 0.35, 0.45]
for snr_sig = -6:2:18
for modulation_idx = 1:length(modulationTypes)
    
    Ls = vLs(sps_idx);
    Ms = vMs(sps_idx);
    sps = Ls / Ms;
    modulation = modulationTypes(modulation_idx);

    if ~isfile(sprintf("../../data/signal/%s/%s_%s_%0.1g_%0.1g_%0.2g_%d.mat", "rx_s", channel, ...
            modulation, fs, sps, RC, snr_sig))
        continue
    end
    fprintf("%s - Computing %s_%s_%0.1g_%0.1g_%0.2g_%d.mat \n", ...
        datestr(toc/86400,'HH:MM:SS'), channel, modulation, fs, sps, RC, snr_sig);
    
    loadfun = @(var) load(sprintf("../../data/signal/%s/%s_%s_%0.1g_%0.1g_%0.2g_%d.mat", var, channel, ...
            modulation, fs, sps, RC, snr_sig), var).(var);
    y = loadfun("y");
    tx_s = loadfun("tx_s");
    rx_s = loadfun("rx_s");
    rx_nf = loadfun("rx_nf");

    if contains(char(modulation), {'GFSK', 'CPFSK', 'B-FM', 'DSB-AM', 'SSB-AM','OQPSK'})
        yml = y;
        yml_nf = y;
        yml_est = y;
    else
        snrs = arrayfun(@(x) snr(tx_s(x,:), rx_s(x,:) - tx_s(x,:)), 1:length(tx_s));
        snrs = transpose(snrs);
        yml = MaximumLikelihood(rx_s, snr_sig, 1);
        yml_nf = MaximumLikelihood(rx_nf, snr_sig, 1);
        yml_est = MaximumLikelihood(rx_s, snrs, 1);
    end
    for var = ["yml", "yml_nf", "yml_est"]
        path = sprintf("../../data/signal/%s/%s_%s_%0.1g_%0.1g_%0.2g_%d.mat", var, channel, ...
            modulation, fs, sps, RC, snr_sig);
        save(path, var);
    end
end
end
end
end
end
end