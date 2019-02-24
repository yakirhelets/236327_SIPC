clear all;
close all;
clc;

%initials constants
N = 64;
c = 0.8;
set_size = 30000;

% caluclate the signals set
set = zeros(N,set_size);
autocorrelation_matrix = zeros(N,N);
for i=1:1:set_size
    K = unidrnd(N/2);
    M = normrnd(0,sqrt(c));
    L = normrnd(0,sqrt((N/2)*(1-c)));

    cur_vec = [repelem(M,(K - 1)) (M + L) repelem(M,(N/2 - 1)) (M + L) repelem(M,(N/2 - K))]';

    set(:,i) = cur_vec;
    autocorrelation_matrix = autocorrelation_matrix + cur_vec * cur_vec';
end
autocorrelation_matrix = autocorrelation_matrix/set_size;
empirical_mean = sum(set,2)/set_size;

% plot the emirical mean signal of the set
figure;
plot(empirical_mean);
title('Empirical Mean of 30,000 signals');
xlabel('element number'); % x-axis label
ylabel('value'); % y-axis label

% plot the autocorrelation matrix
figure;
imagesc(autocorrelation_matrix);
title('Autocorrelation Matrix of set of size 30,000');
colorbar;

% calculate the Wiener filter
W = autocorrelation_matrix * ((autocorrelation_matrix + eye(N))\eye(N));
figure;
imagesc(W);
title('Wiener Filter Matrix');
colorbar;

% calculated noisy signal set
n = normrnd(0,1,N,set_size);
noisy_set = set + n;

% calculate the denisoed set by multiplying the Wiener filter with each
% nosiy signal.
denosied_set = zeros(N,set_size);
for i=1:1:set_size
    denosied_set(:,i) = W * noisy_set(:,i);
end

% plot some signals to compare between the denoised, noisy and clean
% versions
figure;
plot(set(:,1));
hold on;
plot(noisy_set(:,1));
plot(denosied_set(:,1));
legend('clean signal','noisy signal', 'denoised signal');
title('Denoised signals with respect to their clean and the noisy versions #1.');
hold off;

figure;
plot(set(:,floor(set_size/2)));
hold on;
plot(noisy_set(:,floor(set_size/2)));
plot(denosied_set(:,floor(set_size/2)));
legend('clean signal','noisy signal', 'denoised signal');
title('Denoised signals with respect to their clean and the noisy versions #2.');
hold off;

figure;
plot(set(:,set_size));
hold on;
plot(noisy_set(:,set_size));
plot(denosied_set(:,set_size));
legend('clean signal','noisy signal', 'denoised signal');
title('Denoised signals with respect to their clean and the noisy versions #3.');
hold off;

% calculate the empirical mse of the clean and denoised signal
empirical_mse = (sum(sum((set - denosied_set).^2)/N))/set_size;


% again for subsection c

% calculate the Wiener filter
W_2 = autocorrelation_matrix * ((autocorrelation_matrix + 5 * eye(N))\eye(N));
figure;
imagesc(W_2);
title('Wiener Filter Matrix');
colorbar;

% calculated noisy signal set
n_2 = normrnd(0,sqrt(5),N,set_size);
noisy_set_2 = set + n_2;

% calculate the denisoed set by multiplying the Wiener filter with each
% nosiy signal.
denosied_set_2 = zeros(N,set_size);
for i=1:1:set_size
    denosied_set_2(:,i) = W_2 * noisy_set_2(:,i);
end

% plot some signals to compare between the denoised, noisy and clean
% versions
figure;
plot(set(:,1));
hold on;
plot(noisy_set_2(:,1));
plot(denosied_set_2(:,1));
legend('clean signal','noisy signal', 'denoised signal');
title('Denoised signals with respect to their clean and the noisy versions #1.');
hold off;

figure;
plot(set(:,floor(set_size/2)));
hold on;
plot(noisy_set_2(:,floor(set_size/2)));
plot(denosied_set_2(:,floor(set_size/2)));
legend('clean signal','noisy signal', 'denoised signal');
title('Denoised signals with respect to their clean and the noisy versions #2.');
hold off;

figure;
plot(set(:,set_size));
hold on;
plot(noisy_set_2(:,set_size));
plot(denosied_set_2(:,set_size));
legend('clean signal','noisy signal', 'denoised signal');
title('Denoised signals with respect to their clean and the noisy versions #3.');
hold off;

% calculate the empirical mse of the clean and denoised signal
empirical_mse_2 = (sum(sum((set - denosied_set_2).^2)/N))/set_size;
