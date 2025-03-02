load_display_dicom;  % Call the script

A2=regiongrowing(MR,270,430,480);
imshow(A2)
A1=regiongrowing(MR,368,160,480);
imshow(A1)
A3 = A1 + A2;
figure;
imshow(A3)

imshow(MR, [])
SE1 = strel('disk', 8);
A4 = imdilate(A3, SE1);
imshow(A4)
