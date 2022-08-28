% Save states of all modulations: (n_states, iq)
modulationPool = categorical(["BPSK", "QPSK", "8-PSK", ...
    "16-APSK", "32-APSK", "64-APSK", "128-APSK", "256-APSK",...
    "PAM4", "16-QAM", "32-QAM", "64-QAM", "128-QAM", "256-QAM"]);

for j = 1:length(modulationPool)
    modulation = modulationPool(j);

    M = getM(modulation);
    data = 0:1:M-1;
    % The dummy variables are only used on modulations which are not
    % supported
    mod = getModulator(modulation,-1,-1,-1,-1,-1);
    txSig = mod(data);
    states = zeros(length(data), 2);
    states(:,1) = real(txSig);
    states(:,2) = imag(txSig);
    path = sprintf("../../data/signal/states/%s.mat", modulation);
    save(path, "states");
end