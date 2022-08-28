tic

spf = 4096;  % samples per frame
n_signals = 1000;
n_span_sym = 8;
fc_d = 902e6;  % Central frequency for digital modulations
fc_a = 100e6;  % Central frequency for analog modulations
fc_am = 50e3;  % Central frequency for AM
transDelay= 2*n_span_sym;

applyRandomFrame = false;

% Modulations
modulationTypes = categorical(["BPSK", "QPSK", "8-PSK", ...
    "16-APSK", "32-APSK", "64-APSK", "128-APSK", "256-APSK",...
    "PAM4", "16-QAM", "32-QAM", "64-QAM", "128-QAM", "256-QAM", ... 
    "GFSK", "CPFSK", "OQPSK", "B-FM", "DSB-AM", "SSB-AM"]);

% For each signal in mixture, power attenuation is chosen uniform-randomly
% from the given range
% powerRatiosDb = -10:2:10; 

% Clock offset options
applyClockOffset = false;
maxDeltaOff = 5;  % Maximum clock offset of 5 ppm (parts per million)

% Create folders
mkdir('../../data/signal/tx_s');
mkdir('../../data/signal/rx_s');
mkdir('../../data/signal/rx_x');
mkdir('../../data/signal/y');

% Number of samples per symbol
%vLs = [2, 4, 8, 16, 16, 32, 32, 32];
%vMs = [1, 1, 1, 1, 5, 1, 3, 5];
%vLs = [2, 4, 8, 16, 32];
%vMs = [1, 1, 1, 1, 1];
vLs = [8];
vMs = [1];
assert(length(vLs) == length(vMs))

seed = 0;
for channel = ["AWGN"]%, "Rayleigh", "Rician"]
for fs = 0.2e6 %[1e6, 1.5e6, 2e6] %0.2e6, 0.6e6
for sps_idx = 1:length(vLs)
for RC = [0.35]
for snr = 5
for modulation_idx =11
    Ls = vLs(sps_idx);
    Ms = vMs(sps_idx);
    sps = Ls / Ms;
    modulation = modulationTypes(modulation_idx);

    if isfile(sprintf("../../data/signal/%s/%s_%s_%0.2g_%0.2g_%0.2g_%d.mat", "rx_x", channel, ...
            modulation, fs, sps, RC, snr))
        %continue
    end
    fprintf("%s - Generating %s_%s_%0.2g_%0.2g_%0.2g_%d.mat \n", ...
        datestr(toc/86400,'HH:MM:SS'), channel, modulation, fs, sps, RC, snr);

    is_analog = contains(char(modulation), {'B-FM','DSB-AM','SSB-AM'});

    if is_analog
        n_symb = spf;
        fc = fc_a;
    else
        % The frame can be offset by a fraction of symbol. Thus, we only
        % save full symbols
        n_symb = spf/sps;  % Substract one if we do the offsets
        fc = fc_d;
    end
    
    dataSrc = createBitData(modulation, sps, 2*Ms*spf, fs);
    mod = getModulator(modulation, Ls, n_span_sym, RC, fs, fc_am);
    %demod = getDemodulator(modulation, Ls, n_span_sym, RC, fs, fc_a);
    tx_filt = getTxFilt(modulation, Ls, Ms, n_span_sym, RC);
    rx_filt = getRxFilt(modulation, Ls, Ms, n_span_sym, RC);

    tx_s = zeros(n_signals, n_symb);
    rx_s = zeros(n_signals, n_symb);
    rx_x = zeros(n_signals, spf);
    y = zeros(n_signals, length(modulationTypes));
    y(:, modulation_idx) = 1;
    for i = 1:n_signals
        xts = dataSrc();
        xtm = mod(xts);
        xtf = tx_filt(xtm);
        xrf = applyChannel(xtf, channel, fs);
        xrf_awgn = awgn(xtf, snr, 'measured',seed);
        xrf = awgn(xrf, snr, 'measured',seed);
        
        %xrs = demod(xrm);
        % Create frame
        if(applyRandomFrame==true)
            startIdx = randi([transDelay*Ls length(xrf)-transDelay-spf]);
        else
            startIdx = Ls*transDelay;
        end
        rx_x(i,:) = xrf(startIdx+(1:spf),1);
        if ~contains(char(modulation), {'GFSK', 'CPFSK', 'B-FM', 'DSB-AM', 'SSB-AM','OQPSK'})
            %rx_s(i,:) = rx_filt(transpose(rx_x(i,:)));
            xrm = rx_filt(xrf_awgn);
            tx_s(i,:) = xtm(startIdx/Ls+(1:n_symb));
            rx_s(i,:) = xrm(startIdx/Ls+n_span_sym+(1:n_symb));  % There is a delay of n_span_sym btw tx_s and rx_s
            
        else
            tx_s = -1;
            rx_s = -1;
            rx_nf = -1;
        end
        
        % Normalize
        % rx_x(i,:) = rx_x(i,:) / sqrt(sum(abs(rx_x(i,:)).^2));
        
        % Save the rest of variables
        %tx_s(:,i) = xts();
        seed = seed + 1;
    end
    for var = ["tx_s", "rx_x", "rx_s", "y"]
        path = sprintf("../../data/signal/%s/%s_%s_%0.2g_%0.2g_%0.2g_%d.mat", var, channel, ...
            modulation, fs, sps, RC, snr);
        save(path, var);
    end
    
end
end
end
end
end
end