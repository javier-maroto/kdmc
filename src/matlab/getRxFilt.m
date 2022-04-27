function rx_filt = getRxFilt(modulation, Ls, Ms, n_span_sym, rolloff)

switch modulation
    case {'BPSK','QPSK','8-PSK','16-PSK','32-PSK','16-APSK','32-APSK',...
            '64-APSK','128-APSK','256-APSK','16-QAM','32-QAM','64-QAM','128-QAM',...
            '256-QAM','PAM4'}
        rxfilter = comm.RaisedCosineReceiveFilter(...
            'RolloffFactor', rolloff, ...
            'FilterSpanInSymbols', n_span_sym, ...
            'InputSamplesPerSymbol', Ls, 'DecimationFactor', Ls, 'Gain', 1/sqrt(Ls));
        rx_filt = @(x) rxfilter(x);
        if(Ms > 1)
            rrcFilter=rcosdesign(rolloff,n_span_sym,Ms);
            % add gain since due to downsampling the signal amplitude changes
            rx_filt = @(x) upfirdn(rx_filt(x).*Ls,rrcFilter,1,Ms(sps_i));
        end
    case {'OQPSK',"GFSK","CPFSK","B-FM","DSB-AM","SSB-AM"}
        rx_filt = @(x) x;
    otherwise
        error(modulation);
end

end