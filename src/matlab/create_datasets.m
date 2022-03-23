n_samples = 1024;
n_signals = 1000;
n_span_sym = 4;

mkdir('../../data/signal/tx_b');
mkdir('../../data/signal/rx_x');

easy_mods = ["BPSK", "QPSK", "8-PSK", "16-QAM", "64-QAM", "PAM4", ... 
    "GFSK", "CPFSK", "BFM", "DSB-AM", "SSB-AM"];
hard_mods = ["OQPSK", "32-QAM", "128-QAM", "256-QAM", "16-APSK", ...
    "32-APSK", "64-APSK", "128-APSK", "256-APSK"];
for channel = ["AWGN", "Rayleigh", "Rician"]
for modulation = [easy_mods, hard_mods]
for fs = [0.2e6, 0.6e6, 1e6, 1.5e6, 2e6]
for L = [2, 3.2, 4, 6.4, 8, 32/3, 16, 32]
for RC = [0.15, 0.25, 0.35, 0.45]
for snr = -6:18
    if isfile(sprintf("../../data/signal/%s/%s_%s_%0.1g_%d_%0.2g_%d.mat", "tx_b", channel, ...
            modulation, fs, L, RC, snr))
        continue
    end
    n_symb = n_samples / L;
    [symb2sig, M] = getModulator(modulation, L, n_span_sym, RC, fs);
    n_bits = n_symb * log2(M);
    
    tx_b = randi([0 1], n_bits, n_signals);
    tx_x = zeros(n_samples, n_signals);
    rx_x = zeros(n_samples, n_signals);
    for i = 1:n_signals
        tx_x(:,i) = symb2sig(tx_b(:,i));
        switch channel
            case {"AWGN"}
                rx_x(:,i) = awgn(tx_x(:,i), snr, 'measured');
            otherwise
                error(channel)
        end
    end
    for var = ["tx_b", "rx_x"]
        path = sprintf("../../data/signal/%s/%s_%s_%0.1g_%d_%0.2g_%d.mat", var, channel, ...
            modulation, fs, L, RC, snr);
        save(path, var);
    end
end
end
end
end
end
end