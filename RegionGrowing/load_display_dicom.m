function MR = load_display_dicom()
    % Define the file path with a wildcard for selection
    FilePath = ("/MATLAB Drive/*.dcm");  

    % Open file selection dialog
    [dFILENAME, dPATHNAME] = uigetfile(FilePath, 'File Open');  

    % Check if user canceled selection
    if isequal(dFILENAME, 0)
        disp('User canceled file selection.');
        MR = [];  % Return empty if no file is selected
        return;
    end

    % Construct full filename and read DICOM data
    filename = fullfile(dPATHNAME, dFILENAME);
    info = dicominfo(filename);
    MR = dicomread(info);

    % Display the image
    figure, imagesc(MR), title('RAW MR IMAGE');
    figure; imshow(MR, []);

    % Convert to double for further processing
    MR = double(MR);
end

