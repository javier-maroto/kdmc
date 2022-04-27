function M = getM(modulation)
    switch modulation
        case {'BPSK','GFSK',"CPFSK"}
            M = 2;
        case {'QPSK',"PAM4",'OQPSK'}
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
        case {'256-QAM',"256-APSK"}
            M = 256;
        case {"B-FM",'DSB-AM','SSB-AM'}
            M = -1;
        otherwise
            error(modulation)
    end
end

