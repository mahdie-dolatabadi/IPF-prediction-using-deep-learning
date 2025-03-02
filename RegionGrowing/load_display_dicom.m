% Define the file path with a wildcard to allow selection of any file type
FilePath = ("/MATLAB Drive/*.*");  

% Open a file selection dialog and get the chosen file name and path
[dFILENAME, dPATHNAME] = uigetfile(FilePath, 'File Open');  

% Construct the full file path
filename = [dPATHNAME dFILENAME];  

% Read DICOM metadata from the selected file
info = dicominfo(filename);  

% Read the DICOM image data
MR = dicomread(info);  

% Display the raw MR image using imagesc (scales intensity values)
figure, imagesc(MR), title('RAW MR IMAGE');

% Display the image using imshow with automatic intensity scaling
figure; imshow(MR, []);  

% Convert the image data to double precision for further processing
MR = double(MR);
