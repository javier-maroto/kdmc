clear all;
close all;

n_signals = 1000;
sps = 8;
n_sym = 1024;
n_span_sym = 4;
rolloff = 0.35;
snr = 20;
%k = log2(M);
%filtDelay = k * n_span_sym;
%n_bits = k * n_sym;

rx_filt = comm.RaisedCosineReceiveFilter( ...
    'Shape','Normal', ...
    'RolloffFactor', rolloff, ...
    'FilterSpanInSymbols', n_span_sym, ...
    'InputSamplesPerSymbol', sps, ...
    'DecimationFactor', sps);

constDiagram = comm.ConstellationDiagram( ...
    'SamplesPerSymbol', sps, ...
    'SymbolsToDisplaySource', 'Property', ...
    'SymbolsToDisplay', n_sym - n_span_sym, ...
    'ShowReferenceConstellation', false);

to_complex = @(x) squeeze(x(:,1,:) + 1i*x(:,2,:))';
%% Standardly trained model

% Assumes we are in this file folder
load_path = '../../../logs/custom.rml2018/VT_CNN2_BF/Std/version_1/results/adv_matlab/x_adv.mat';
load(load_path)

x = to_complex(x);
x_adv = to_complex(x_adv);
k = 0;
%% 
% BPSK example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% QPSK example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% 8PSK example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% 16PSK example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% 32PSK example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% 16APSK example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% 32APSK example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% 64APSK example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% 128APSK example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% 16QAM example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% 32QAM example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% 64QAM example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% 128QAM example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% 256QAM example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% OQPSK example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% Adversarially trained model

% Assumes we are in this file folder
load_path = '../../../logs/custom.rml2018/VT_CNN2_BF/AT/version_1/results/adv_matlab/x_adv.mat';
load(load_path)

x = to_complex(x);
x_adv = to_complex(x_adv);
k = 0;
%% 
% BPSK example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% QPSK example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% 8PSK example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% 16PSK example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% 32PSK example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% 16APSK example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% 32APSK example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% 64APSK example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% 128APSK example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% 16QAM example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% 32QAM example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% 64QAM example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% 128QAM example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% 256QAM example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%% 
% OQPSK example

i = find(y_adv == k, 1, 'first');
constDiagram([x(:,i), x_adv(:,i)])
labels(y_adv(i) + 1, :)
labels(pred_adv(i) + 1, :)
k = k + 1;
%%
% for i = fails
%     constDiagram([x(:,i), x_adv(:,i)])
%     pause(0.2)
% end
% 
% tx_b = demod(rx_filt(x));
% errorStats = errorRate(tx_b, rx_b);
%
%
% figure;
% plot(real(tx_x(filtDelay:end)),'b');
% hold on
% plot(imag(tx_x(filtDelay:end)),'g');
% legend('Inphase signal', 'Quadrature signal');
%
% figure;
% plot(real(rx_x(filtDelay:end)),'b');
% hold on
% plot(imag(rx_x(filtDelay:end)),'g');
% legend('Inphase signal', 'Quadrature signal');