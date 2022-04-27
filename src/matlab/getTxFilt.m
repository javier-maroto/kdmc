function tx_filt = getTxFilt(modulation, Ls, Ms, n_span_sym, rolloff)

switch modulation
    case {'BPSK','QPSK','8-PSK','16-PSK','32-PSK','16-APSK','32-APSK',...
            '64-APSK','128-APSK','256-APSK','16-QAM','32-QAM','64-QAM','128-QAM',...
            '256-QAM','PAM4'}
        txfilter = comm.RaisedCosineTransmitFilter(...
            'RolloffFactor', rolloff, ...
            'FilterSpanInSymbols', n_span_sym, ...
            'OutputSamplesPerSymbol', Ls, 'Gain', sqrt(Ls));
        tx_filt = @(x) txfilter(x);
        if(Ms > 1)
            rrcFilter=rcosdesign(rolloff,n_span_sym,Ms);
            % add gain since due to downsampling the signal amplitude changes
            tx_filt = @(x) upfirdn(tx_filt(x).*Ls,rrcFilter,1,Ms);
        end
    case {'OQPSK',"GFSK","CPFSK","B-FM","DSB-AM","SSB-AM"}
        tx_filt = @(x) x;
    otherwise
        error(modulation);
end

end