close all;
f=imread('lena.png'); %image read
figure('name','Clean image');
imshow(f);   %image display
%%% σn = 8, a) σb = 0.5,%%%
SDb=0.5;
SDn=8;
%Blurring the image
[g,h]=imageblur(SDb,f,SDn);
figure('name','Blurred image σn = 8,  σb = .5');
%Displaying the blurred image
imshow(uint8(g));
% vary λ from 0.01 to 2.0 in steps of 0.01 and pick the λ that gives minimum RMS
%error between the original image and the estimated image
[Deblurred,RmseMin]=findminRMSEImage(g,f,h);
figure('name','DeBlurred image σn = 8,  σb = 0.5');
%Displaying the deblurred image
imshow(uint8(Deblurred));
%Visually good DeBlurred image
lambda=.15;
Deblurred=uint8(imageDeblur(g,h,lambda));
figure('name','Visually good DeBlurred image σn = 8,  σb = 0.5');
imshow(uint8(Deblurred));

%%% σn = 8, a) σb = 1,%%%
SDb=1;
SDn=8;
%Blurring the image
[g,h]=imageblur(SDb,f,SDn);
figure('name','Blurred image σn = 8,  σb = 1');
imshow(uint8(g));
% vary λ from 0.01 to 2.0 in steps of 0.01 and pick the λ that gives minimum RMS
%error between the original image and the estimated image
[Deblurred,RmseMin]=findminRMSEImage(g,f,h);
figure('name','DeBlurred image σn = 8,  σb = 1');
imshow(uint8(Deblurred));
lambda=.07;
Deblurred=uint8(imageDeblur(g,h,lambda));
figure('name','Visually good DeBlurred image σn = 8,  σb = 1');
imshow(uint8(Deblurred));

%%% σn = 8, a) σb = 1.5,%%%
SDb=1.5;
SDn=8;
%Blurring the image
[g,h]=imageblur(SDb,f,SDn);
figure('name','Blurred image σn = 8,  σb = 1.5');
imshow(uint8(g));
% vary λ from 0.01 to 2.0 in steps of 0.01 and pick the λ that gives minimum RMS
%error between the original image and the estimated image
[Deblurred,RmseMin]=findminRMSEImage(g,f,h);
figure('name','DeBlurred image σn = 8,  σb = 1.5');
imshow(uint8(Deblurred));
lambda=.05;
Deblurred=uint8(imageDeblur(g,h,lambda));
figure('name','Visually good DeBlurred image σn = 8,  σb = 1.5');
imshow(uint8(Deblurred));

%%% σn = 5, a) σb = 1,%%%
SDb=1;
SDn=5;
%Blurring the image
[g,h]=imageblur(SDb,f,SDn);
figure('name','Blurred image σn = 5, a) σb = 1');
imshow(uint8(g));
% vary λ from 0.01 to 2.0 in steps of 0.01 and pick the λ that gives minimum RMS
%error between the original image and the estimated image
[Deblurred,RmseMin]=findminRMSEImage(g,f,h);
figure('name','DeBlurred image σn = 5, a) σb = 1');
imshow(uint8(Deblurred));
lambda=.03;
Deblurred=uint8(imageDeblur(g,h,lambda));
figure('name','Visually good DeBlurred image σn = 5,  σb = 1');
imshow(uint8(Deblurred));

%%% σn = 10, a) σb = 1,%%%
SDb=1;
SDn=10;
%Blurring the image
[g,h]=imageblur(SDb,f,SDn);
figure('name','Blurred image σn = 10, a) σb = 1');
imshow(uint8(g));
% vary λ from 0.01 to 2.0 in steps of 0.01 and pick the λ that gives minimum RMS
%error between the original image and the estimated image
[Deblurred,RmseMin]=findminRMSEImage(g,f,h);
figure('name','DeBlurred image σn = 10, a) σb = 1');
imshow(uint8(Deblurred));
lambda=.03;
Deblurred=uint8(imageDeblur(g,h,lambda));
figure('name','Visually good DeBlurred image σn = 10,  σb = 1');
imshow(uint8(Deblurred));

%%% σn = 15, a) σb = 1,%%%
SDb=1;
SDn=15;
%Blurring the image
[g,h]=imageblur(SDb,f,SDn);
figure('name','Blurred image σn = 15, a) σb = 1');
imshow(uint8(g));
% vary λ from 0.01 to 2.0 in steps of 0.01 and pick the λ that gives minimum RMS
%error between the original image and the estimated image
[Deblurred,RmseMin]=findminRMSEImage(g,f,h);
figure('name','DeBlurred image σn = 15, a) σb = 1');
imshow(uint8(Deblurred));
lambda=.02;
Deblurred=uint8(imageDeblur(g,h,lambda));
figure('name','Visually good DeBlurred image σn = 15,  σb = 1');
imshow(uint8(Deblurred));

%%%L2 Vs L1 regularization %%%
%%%  σn = 1, σb = 1.5 %%%
SDb=1.5;
SDn=1;
%Blurring the image
[g,h]=imageblur(SDb,f,SDn);
figure('name','Blurred image σn = 1, σb = 1.5');
imshow(uint8(g));
lambda=0.01;
%L1 gradient regularization
Deblurred=uint8(admmfft(g,h,lambda,1));
figure('name','DeBlurred image σn = 1, σb = 1.5 L1');
imshow(uint8(Deblurred));
Deblurred=imageDeblur(g,h,lambda);
figure('name','DeBlurred image σn = 1, σb = 1.5 L2');
imshow(uint8(Deblurred));

%%%  σn = 5, σb = 1.5 %%%
SDb=1.5;
SDn=5;
%Blurring the image
[g,h]=imageblur(SDb,f,SDn);
figure('name','Blurred image σn = 1, σb = 1.5');
imshow(uint8(g));
lambda=0.01;
%L1 gradient regularization
Deblurred=uint8(admmfft(g,h,lambda,1));
figure('name','DeBlurred image σn = 5, σb = 1.5 L1');
imshow(uint8(Deblurred));
Deblurred=imageDeblur(g,h,lambda);
figure('name','DeBlurred image σn = 5, σb = 1.5 L2');
imshow(uint8(Deblurred));

%%%  σn = 5 %%%
h=imread('mb-kernel.png');
SDn=5;
[row,col]=size(f);
g=(conv2(f,h,'same')); 
n = SDn*randn(row,col);
g = g+n;
figure('name','Blurred image σn = 5');
imshow(uint8(g));
lambda=1;
%L1 gradient regularization
Deblurred=uint8(admmfft(g,h,lambda,1));
figure('name','DeBlurred image σn = 5 L1');
imshow(uint8(Deblurred));
lambda=0.01;
Deblurred=imageDeblur(g,h,lambda);
figure('name','DeBlurred image σn = 5 L2');
imshow(uint8(Deblurred));



%Function for blurring the orginal image
function [blurred,kernal]=imageblur(SDb,f,SDn)
kernal=GaussianKernal(SDb);                %%Find Gaussian kernal
g=(conv2(f,kernal,'same'));       %%Convolving
[row,col]=size(g);
n = SDn*randn(row,col);  %Adding Gaussian noise
blurred = g+n;
end

function deblurred=imageDeblur(g,h,lambda)
[row,col]=size(g);
qx = [1 -1];            %gradient operators qx = [1 -1] and qy = qxT
qy=transpose(qx);
%where H, Qx, Qy, and G are the 2D Fourier transforms 
%(of the same size as that of g) of h,qx, qy, and g
G=fft2(g,row,col);
QX=fft2(qx,row,col);
QY=fft2(qy,row,col);
H=psf2otf(h,[row,col]);
Fcap=(conj(H)./((conj(H).*H)+lambda*(conj(QX).*QX)+lambda*(conj(QY).*QY))).*G;
deblurred=ifft2(Fcap);
end
function RMSE=findrmse(y,yhat)
RMSE = sqrt(mean((y - yhat).^2,'all'));
end

%function for vary λ from 0.01 to 2.0 in steps of 0.01 and pick the λ that gives minimum RMS
%error between the original image and the estimated image.
function [Deblurred,RmseMin]=findminRMSEImage(g,f,h)
lambda=0; %initializing lambda
[row,col]=size(g); %Finding the size of image
deblurred=zeros(row,col,200);
RMSE=zeros(1,200);
%varying lamda and finding rmse 
for i=1:200
    
    lambda=lambda+0.01;
    deblurred(:,:,i)=uint8(imageDeblur(g,h,lambda));
    RMSE(i)=findrmse(uint8(f),uint8(deblurred(:,:,i)));
    
end
%Finding the minimum rmse
[RmseMin,index]=min(RMSE);
Deblurred=deblurred(:,:,index);
end






