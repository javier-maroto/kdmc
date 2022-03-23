function [symb2sig, M] = getModulator(modulation, sps, n_span_sym, rolloff, fs)
%GETMODULATOR returns modulators and demodulators
%   TODO: OOK, AM-SSB-SC, AM-SSB-WC, AM-DSB-SC, AM-DSB-WC, correct 4ASK,
%   8ASK, use audio signal for FM

tx_filt = comm.RaisedCosineTransmitFilter( ...
    'Shape','Normal', ...
    'RolloffFactor', rolloff, ...
    'FilterSpanInSymbols', n_span_sym, ...
    'OutputSamplesPerSymbol', sps, ...
    'Gain', sqrt(sps));

switch modulation
    case {'BPSK','GMSK'}
        M = 2;
    case {'QPSK','OQPSK'}
        M = 4;
    case {'8-PSK'}
        M = 8;
    case {'16-PSK','16-APSK','16-QAM'}
        M = 16;
    case {'32-PSK','32-APSK','32-QAM'}
        M = 32;
    case {'64-APSK','64-QAM'}
        M = 64;
    case {'128-APSK','128-QAM'}
        M = 128;
    case {'256-QAM'}
        M = 256;
    otherwise
        error(modulation)
end

switch modulation
    case {'BPSK','8-PSK','16-PSK','32-PSK'}
        mod = comm.PSKModulator(...
            'ModulationOrder', M, 'PhaseOffset', 0, 'BitInput', true);
    case {'QPSK'}
        mod = comm.PSKModulator(...
            'ModulationOrder', M, 'PhaseOffset', pi/M, 'BitInput', true);
    case {'16-APSK','32-APSK'}
        mod = @(x) dvbsapskmod(...
            x, M, 's2', 'UnitAveragePower', true, 'InputType', 'bit');
    case {'64-APSK','128-APSK'}
        mod = @(x) dvbsapskmod(...
            x, M, 's2x', 'UnitAveragePower', true, 'InputType', 'bit');
    case {'16-QAM','32-QAM','64-QAM','128-QAM','256-QAM'}
        mod = @(x) qammod(...
            x, M, 'gray', 'UnitAveragePower', true, 'InputType', 'bit');
    case {'OQPSK'}
        mod = comm.OQPSKModulator(...
            'PulseShape', 'Normal raised cosine', ...
            'RolloffFactor', rolloff, ...
            'SamplesPerSymbol', sps, 'SymbolMapping', 'Gray', ...
            'FilterSpanInSymbols', n_span_sym, 'BitInput', true);
    case {'GMSK'}
        mod = comm.GMSKModulator(...
            'BitInput', true, 'SamplesPerSymbol', sps);
end

switch modulation
    case {'BPSK','QPSK','8-PSK','16-PSK','32-PSK','16-APSK','32-APSK',...
            '64-APSK','128-APSK','16-QAM','32-QAM','64-QAM','128-QAM',...
            '256-QAM'}
        symb2sig = @(x) tx_filt(mod(x));
    case {'OQPSK','GMSK'}
        symb2sig = @(x) mod(x);
end


