tic

spf = 1024;  % samples per frame
v_snr = -20:2:30;
Ls = 8;
Ms = 1;
sps = 8;
v_rolloff = [0.2];
v_ray_delay = [0, 0.5, 1.0, 2.0];
sym_delay = 0:1:16;
fs = 200e6;
clockOff = 0/1e6;


n_signals = 1000000;
n_span_sym = 8;
fc_d = 902e6;  % Central frequency for digital modulations
fc_a = 100e6;  % Central frequency for analog modulations
fc_am = 50e3;  % Central frequency for AM
transDelay= 2*n_span_sym;

%L_veh = 21;
% L_ped = 11;
%v1 = 4 * 1e3 / 3600; % Mobile speed (m/s)
% v2 = 100 * 1e3 / 3600; % Mobile speed (m/s)
%fc = 2e+9; % Carrier frequency
%c = physconst('LightSpeed'); % Speed of light in free space
%max_doppler_shift1 = v1*fc/c; 

%for i=1:num_channels
%ch_resp_ped = stdchan(1/fs, max_doppler_shift1, 'itur3GVAx');
%ch_resp_ped.StoreHistory=1;
%x=zeros(1,L_ped);
%x(1)=1;
%y_veh(i,:)=filter(ch_resp_veh,x);
%end


% Modulations
modulationTypes = categorical(["BPSK", "QPSK", "8-PSK", "16-PSK", "32-PSK", ...
    "16-APSK", "32-APSK", "64-APSK", "128-APSK", ...
    "16-QAM", "32-QAM", "64-QAM", "128-QAM", "256-QAM"]);

% For each signal in mixture, power attenuation is chosen uniform-randomly
% from the given range
% powerRatiosDb = -10:2:10; 

randElem = @(x) x(randi(length(x)));

seed = 0;
tx_s = zeros(n_signals, spf);
rx_s = zeros(n_signals, spf);
rx_x = zeros(n_signals, spf);
y = zeros(n_signals, length(modulationTypes));
snrs = [];
snrs_filt = [];
lsps = [];
rolloffs = [];
fss = [];
rays = [];
for i = 1:n_signals
    fprintf("%s - %d\n", datestr(toc/86400,'HH:MM:SS'), i);
    modulation_idx = randi(length(modulationTypes));
    modulation = modulationTypes(modulation_idx);
    RC = randElem(v_rolloff);
    snr_i = randElem(v_snr);
    ray_delay = randElem(v_ray_delay);
    fs_offset = fs * randn()*clockOff;

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

    fc = fc * (1 + randn()*clockOff);
    
    dataSrc = createBitData(modulation, sps, 2*Ms*spf, fs);
    mod = getModulator(modulation, Ls, n_span_sym, RC, fs, fc_am);
    tx_filt = getTxFilt(modulation, Ls, Ms, n_span_sym, RC);
    rx_filt = getRxFilt(modulation, Ls, Ms, n_span_sym, RC);

    y(i, modulation_idx) = 1;
    
    x = dataSrc();
    xt = mod(x);
    x = tx_filt(xt);
    xml = awgn(x, snr_i, 'measured', seed);
    
    pfo = comm.PhaseFrequencyOffset('SampleRate',fs,'FrequencyOffset',fs_offset);
    x=pfo(x);
    rayChan = comm.RayleighChannel(...
        'SampleRate',fs, ...
        'PathDelays', ray_delay/fs, ...
        'NormalizePathGains',true, ...
        'MaximumDopplerShift',0.001, ...
        'PathGainsOutputPort',true);
    x=rayChan(x);
%     x = filter(ch_resp_ped,x);
    x = awgn(x, snr_i, 'measured', seed);
        
    % Create frame
    startIdx = Ls*transDelay + randElem(sym_delay);
    x = x(startIdx+(1:spf),1);
    rx_x(i,:) = x / sqrt(sum(abs(x).^2));
    if ~contains(char(modulation), {'GFSK', 'CPFSK', 'B-FM', 'DSB-AM', 'SSB-AM','OQPSK'})
        xml = rx_filt(xml);
        tx_s(i,1:n_symb) = xt(floor(startIdx/Ls)+(1:n_symb));
        rx_s(i,1:n_symb) = xml(floor(startIdx/Ls)+n_span_sym+(1:n_symb));  % There is a delay of n_span_sym btw tx_s and rx_s
        snr_filt = snr(tx_s(i,1:n_symb), rx_s(i,1:n_symb) - tx_s(i,1:n_symb));
        %assert(snr_filt > snr_i)
    end
    seed = seed + 1;
    snrs = [snrs, snr_i];
    snrs_filt = [snrs_filt, snr_filt];
    lsps = [lsps, sps];
    rolloffs = [rolloffs, RC];
    fss = [fss, fs];
    rays = [rays, ray_delay];
end
tmp = zeros(size(rx_x,1), 2, size(rx_x,2));
tmp(:,1,:) = real(rx_x);
tmp(:,2,:) = imag(rx_x);
rx_x = tmp;
tmp = zeros(size(rx_s,1), 2, size(rx_s,2));
tmp(:,1,:) = real(rx_s);
tmp(:,2,:) = imag(rx_s);
rx_s = tmp;
save("../../data/signal/srml2018_4.mat", "rx_x", "rx_s", "y", "snrs", "snrs_filt", "lsps", "rolloffs", "fss", "rays", "-v7.3");
