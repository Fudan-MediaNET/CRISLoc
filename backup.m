clear;
trNum = 1;
reNum = 3;
chaNum = 30;

% averages_cross_channel = zeros(rowGrid,colGrid,trNum,reNum);
% averages = zeros(rowGrid,colGrid,trNum,reNum,chaNum);
% variances = zeros(rowGrid,colGrid,trNum,reNum,chaNum);

csi_trace = read_bf_file('csimoving.dat');
len = length(csi_trace);
csi_list = zeros(len, trNum, reNum, chaNum);
for j = 1:len
    csi_entry = csi_trace{j};
    csi = get_scaled_csi(csi_entry);
    csi_list(j,:,:,:) = csi;
end
% Average
csi_ave = sum(abs(csi_list),1)/len;
csi_ave = reshape(csi_ave,[trNum, reNum, chaNum]);
csi_ave_cross_channel = sum(csi_ave,3)/chaNum;
csi_ave_cross_channel = reshape(csi_ave_cross_channel,[trNum, reNum]);

% Variance
csi_abs = abs(csi_list);
csi_var = var(csi_abs,0,1);
csi_var = reshape(csi_var,[trNum, reNum, chaNum]);

