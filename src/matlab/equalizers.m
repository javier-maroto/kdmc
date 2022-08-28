eq = comm.LinearEqualizer("Algorithm", "LMS");
%eq = comm.DecisionFeedbackEqualizer("Algorithm","RLS");
eq.Constellation = pskmod(0:1,2,0);
ms = maxstep(eq, xrf);
[rx_eq, err] = eq(xrf, xtf);
%rx_eq2 = rx_filt(rx_eq);
scatterplot(xrf)
scatterplot(rx_eq)