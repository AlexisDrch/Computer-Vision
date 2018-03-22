
clc;

img1 = im2double(imread('Shift0.png'));
img2 = im2double(imread('ShiftR2.png'));

img1 = imfilter(img1, fspecial('gaussian', 100,4));
img2 = imfilter(img2, fspecial('gaussian', 100,4));

Ix = imfilter(img1, [-1 1]);
Iy = imfilter(img1, [-1 1]');

Ix2 = Ix .^ 2;
Iy2 = Iy .^ 2;
Ixy = Ix .* Iy;
It  = img2 - img1;
Ixt = Ix .* It;
Iyt = Iy .* It;


wIx2 = imfilter(Ix2, fspecial('gaussian', 50, 14));
wIy2 = imfilter(Iy2, fspecial('gaussian', 50, 14));
wIxy = imfilter(Ixy, fspecial('gaussian', 50, 14));
wIxt = imfilter(Ixt, fspecial('gaussian', 50, 14));
wIyt = imfilter(Iyt, fspecial('gaussian', 50, 14));

UV = zeros(size(img1,1), size(img1,2), 2);

for y=1:size(img1,1)
    for x=1:size(img1,2)
%         x = 150;
%         y = 120;
        A = [wIx2(y,x) wIxy(y,x); wIxy(y,x) wIy2(y,x)];
        b = [-wIxt(y,x); -wIyt(y,x)];
        uv = A \ (b);
        
        UV(y,x,1) = uv(1);
        UV(y,x,2) = uv(2);
    end
end

%%
subplot(2,1,1); 
imagesc(UV(:,:,1))
colorbar;
subplot(2,1,2);
imagesc(UV(:,:,2))
colorbar;





