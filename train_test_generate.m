function [train_x, test_x, train_y, test_y] = train_test_generate(options)
% INPUT
% options: selection of features

% Constant Variable
trNum = 1;
reNum = 3;
chaNum = 30;
% rowGrid = 6;
% colGrid = 6;
file_path = './data';
train_recods = 1000;
test_records = 100;

switch options
    case 'default'
        % featureNum: Dimension of the feature space
        featureNum = chaNum * reNum * trNum;
    case 'difference'
        featureNum = (2*chaNum-1) * reNum * trNum;
        difference_index = [];
        for k = 1:reNum*trNum
            a = (k-1)*chaNum+2;
            b = k*chaNum;
            difference_index = cat(2,difference_index,[a:b]);
        end
    case 'differenceOnly'
        featureNum = (chaNum-1)*reNum*trNum;
        difference_index = [];
        for k = 1:reNum*trNum
            a = (k-1)*chaNum+2;
            b = k*chaNum;
            difference_index = cat(2,difference_index,[a:b]);
        end
    case 'cal_phase'
        featureNum = chaNum * reNum * trNum;
    otherwise
        error('Options not applicable.')
end

files = dir(file_path);
file_names = {files.name};
len_files = length(file_names);

train_x = zeros(1000000, featureNum);
test_x = zeros(1000000, featureNum);
train_y = zeros(1000000, 1);
test_y = zeros(1000000, 1);

train_counter = 1;
test_counter = 1;
for i = 1:len_files
    temp = char(file_names(i));
    expr = 'csi[1-6]0[1-6].dat';
    if regexp(temp,expr) == 1
        % Label of the position
        pos_lab = str2double(temp(4:6));
        csi_trace = read_bf_file(['data/',temp]);
        len = length(csi_trace);
        % For records in each reference points, regard [10001:10001+train_records] as the 
        % training sets, and the last test_records as the test sets
        for j = 10001:10000 + train_recods
            csi_entry = csi_trace{j};
            csi = get_scaled_csi(csi_entry);
            csi_transposed = permute(abs(csi),[3,1,2]); %[chaNum * trNum * reNum]
            csi_reshaped = reshape(csi_transposed, [1,trNum * reNum * chaNum]);
            switch options
                case 'default'
                    train_x(train_counter,:) = csi_reshaped;
                case 'difference'
                    csi_padded = cat(2,csi_reshaped,0);
                    csi_shifted = cat(2,0,csi_reshaped);
                    difference = csi_shifted - csi_padded;
                    difference = difference(difference_index);
                    train_x(train_counter,:) = cat(2,csi_reshaped,difference);
                case 'cal_phase'
                    train_x(train_counter,1:30) = cal_phase(csi(1,1,:));
                    train_x(train_counter,31:60) = cal_phase(csi(1,2,:));
                    train_x(train_counter,61:90) = cal_phase(csi(1,3,:));
            end
            train_y(train_counter,1) = pos_lab;
            train_counter = train_counter + 1;
        end
        for j = len - test_records +1 :len
            csi_entry = csi_trace{j};
            csi = get_scaled_csi(csi_entry);
            csi_transposed = permute(abs(csi),[3,1,2]); %[chaNum * trNum * reNum]
            csi_reshaped = reshape(csi_transposed, [1,trNum * reNum * chaNum]);
            switch options
                case 'default'
                    test_x(test_counter,:) = csi_reshaped;
                case 'difference'
                    csi_padded = cat(2,csi_reshaped,0);
                    csi_shifted = cat(2,0,csi_reshaped);
                    difference = csi_shifted - csi_padded;
                    difference = difference(difference_index);
                    test_x(test_counter,:) = cat(2,csi_reshaped,difference);
                case 'cal_phase'
                    test_x(test_counter,1:30) = cal_phase(csi(1,1,:));
                    test_x(test_counter,31:60) = cal_phase(csi(1,2,:));
                    test_x(test_counter,61:90) = cal_phase(csi(1,3,:));
                case 'differenceOnly'
                    csi_padded = cat(2,csi_reshaped,0);
                    csi_shifted = cat(2,0,csi_reshaped);
                    difference = csi_shifted - csi_padded;
                    difference = difference(difference_index);
                    test_x(test_counter,:) = difference;
            end
            test_y(test_counter,:) = pos_lab;
            test_counter = test_counter + 1;
        end

%         csi_list = zeros(len, trNum, reNum, chaNum);
%         for j = 1:len
%             csi_entry = csi_trace{j};
%             csi = get_scaled_csi(csi_entry);
%             csi_list(j,:,:,:) = csi;
%         end
%         % Average
%         csi_ave = sum(abs(csi_list),1)/len;
%         csi_ave = reshape(csi_ave,[trNum, reNum, chaNum]);
%         averages(rowIndex, colIndex,:,:,:) = csi_ave;
%         csi_ave_cross_channel = sum(csi_ave,3)/chaNum;
%         csi_ave_cross_channel = reshape(csi_ave_cross_channel,[trNum, reNum]);
%         averages_cross_channel(rowIndex, colIndex,:,:) = csi_ave_cross_channel;
%         % Variance
%         csi_abs = abs(csi_list);
%         csi_var = var(csi_abs,0,1);
%         csi_var = reshape(csi_var,[trNum, reNum, chaNum]);
%         variances(rowIndex, colIndex,:,:,:) = csi_var;
    end
end
train_x = train_x(1:train_counter-1,:);
test_x = test_x(1:test_counter-1,:);
train_y = train_y(1:train_counter-1,1);
test_y = test_y(1:test_counter-1,1);