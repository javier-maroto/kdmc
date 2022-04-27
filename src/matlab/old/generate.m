function [tx_b, tx_x, rx_x, ber] = generate(n_symb, n_signals, modulation, snr, sps, RC, fs)
%GENERATE Creates IQ signals of the specified modulation
%   n_symb: number of symbols per signal
%   n_signals: number of signals
%   modulation: modulation label (e.g. 'BPSK')
%   snr: AWGN SNR added to the signal
%   config_params: [Samples per symbol, FilterSpanInSymbols, RolloffFactor, AudioSampleRate]
arguments
    n_symb
    n_signals
    modulation
    snr
    sps
    RC
    fs
end

[symb2sig, sig2symb, M, errorRate] = getModulator(...
    modulation, [sps, 4, RC, fs]);

n_bits = n_symb * log2(M);
tx_b = randi([0 1], n_bits, n_signals);

tx_x = zeros(n_symb * sps, n_signals);
rx_x = zeros(n_symb * sps, n_signals);
rx_b = zeros(n_bits, n_signals);
for i = 1:n_signals
    tx_x(:,i) = symb2sig(tx_b(:,i));
    rx_x(:,i) = awgn(tx_x(:,i), snr, 'measured');
    rx_b(:,i) = sig2symb(rx_x(:,i));
end

errorStats = zeros(3, n_signals);
for i = 1:n_signals
    errorStats(:,i) = errorRate(tx_b(:,i), rx_b(:,i));
    reset(errorRate);
end

ber = errorStats(1,:);
end

