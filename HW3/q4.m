clear all;
close all;
clc;

%% subsection a

% load an image of resolution 512x512 (colorful)
img = imread('colorful.jpg');
% turn it to a grayscale image so it can be represented as a 2d matrix
gray = rgb2gray(img);
% save the dimentions of the image - m x n
[m,n] = size(gray);
% create a deteriorated version of the grayscale image
deteriorated_const = 42; % we define a constant
deteriorated_gray = gray;
deteriorated_offst = 16;
deteriorated_gray(:,1:deteriorated_offst:end) = deteriorated_gray(:,1:deteriorated_offst:end) + deteriorated_const;
% present the original and the deteriorated grayscale versions of the image
figure;
imshow(gray);
title('Original Grayscale Image');
figure;
imshow(deteriorated_gray);
title('Deteriorated Grayscale Image');

%% subsection b
% we will now show the DFT interference of a line in the image.
% as explain in the report the interference vector can be represented as
% follows:

% let's first define the parameters of the interference in the same
% notations as in question 3.
N = m;
T = deteriorated_offst;
c = N / T;
% now lets build the interference vector.
interference = zeros(m,1);
interference(1:c:end) = (c / sqrt(m)) * deteriorated_const;

%% subsection c
% in this section we will restore the image from its corrupted condition
% by applying notch filtering in the DFT domain for each row.

% DFT matrix
DFT = 1/sqrt(n) * exp(-1i * 2 * pi / N).^((0:(N-1))' * (0:(N-1)));

% create a DFT representation of a row in the deteriorated grayscale image.
dft_deteriorated_gray_256 = abs(DFT * double(deteriorated_gray(N/2,:)'));
% plot this DFT domain signal
figure;
plot(1:N, dft_deteriorated_gray_256);
title('Spectrum of line 256 in deteriorated grayscale image. interferences are marked with *');
xlabel('N samples') % x-axis label
ylabel('Signal in DFT domain') % y-axis label
hold on;
plot(1:c:N, dft_deteriorated_gray_256(1:c:N), 'r*'); % add the markers
hold off;

% create a notch filtered version of the DFT signal of row 256
dft_notched_gray_256 = dft_deteriorated_gray_256;
dft_notched_gray_256(1:c:N) = 0;
% plot the notch filtered signal
figure;
plot(1:N, dft_notched_gray_256);
title('Spectrum of line 256 after Notch filtering of the deteriorated grayscale image');
xlabel('N samples') % x-axis label
ylabel('Notched signal in DFT domain') % y-axis label

% now lets do it for every row of the deteriorated image to produce an
% approximation of the original image.
restored = zeros(m,n);
for i = 1:m
   % first lets compute the signal in the DFT domain
   % dft_signal = DFT * double(deteriorated_gray(i,:)');
   dft_signal = DFT * double(deteriorated_gray(i,:)');
   % now, lets apply notch filterig on the deteriorated points
   % we do not start from the zero index since it represents since it
   % represents the mean graylevel of the entire image
   dft_signal((c+1):c:end) = 0;
   % now lets retun to the image domain by applying conjucate transpose on
   % the DFT and multiply the filtered signal. Save in in the restored
   % matrix. here we are also omitting the imaginary part.
   restored(i,:) = real(DFT' * dft_signal)';
end
% this conversion is to make the next operations work
restored = uint8(restored);
% plot the restored image
figure;
imshow(restored);
title('Grayscale Image restored after applying Notch Filtering');

% to check the MSE between the original image to the restored, we will
% return to the MSE definition:
mse = 1/(m * n) * sum(sum((gray - restored).^2));
