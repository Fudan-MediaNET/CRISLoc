function [accuracy, error_dis] = train_and_test(train_x, test_x, train_y, test_y, options)
trNum = 1;
reNum = 3;
chaNum = 30;

test_len = length(test_y);
train_len = length(train_y);
predicted_y = zeros(test_len, 1);
counter = 0;
error_dis = 0;
switch options
    case 'default'
        for i = 1:test_len
            temp = repmat(test_x(i,:), train_len, 1);
            dist = abs(temp - train_x);
            dist = sum(dist.^2, 2);
            [~,I] = sort(dist);
            neighbours = train_y(I(1:10));
            predict = mode(neighbours);
            predicted_y(i) = predict;
        end
    case 'correlation'
        for i = 1:test_len
            temp = test_x(i,:).';
            cor = train_x*temp;
            [~,I] = sort(cor,'descend');
            neighbours = train_y(I(1:10));
            predict = mode(neighbours);
            predicted_y(i) = predict;
        end
    case 'weight'
        test_x_abs = normalize(test_x(:,1:reNum * trNum * chaNum),2,'norm',2);
        test_x_dif = normalize(test_x(:,reNum * trNum * chaNum+1:(2*chaNum-1) * reNum * trNum),2,'norm',2);
        test_x_normalized = cat(2,test_x_abs, test_x_dif);
        train_x_abs = normalize(train_x(:,1:reNum * trNum * chaNum),2,'norm',2);
        train_x_dif = normalize(train_x(:,reNum * trNum * chaNum+1:(2*chaNum-1) * reNum * trNum),2,'norm',2);
        train_x_normalized = cat(2,train_x_abs, train_x_dif);
        for i = 1:test_len
            temp = repmat(test_x_normalized(i,:), train_len, 1);
            dist = abs(temp - train_x_normalized);
            dist = sum(dist.^2, 2);
            [~,I] = sort(dist);
            neighbours = train_y(I(1:10));
            predict = mode(neighbours);
            predicted_y(i) = predict;
        end
    case 'SVM'
        
    otherwise
        error('Options not applicable.')
end
for i = 1:test_len
    if predicted_y(i) ~= test_y(i)
        counter = counter + 1;
        row_error = abs(floor(predicted_y(i)/100) - floor(test_y(i)/100));
        col_error = abs(mod(predicted_y(i),10) - mod(test_y(i),10));
        error_dis = error_dis + (row_error.^2 + col_error.^2).^0.5;
    end
end
accuracy = (test_len-counter) / test_len;
error_dis = error_dis / test_len;
end