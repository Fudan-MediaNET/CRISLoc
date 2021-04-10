clear;

% accuracy = 0.2575, error_dis = 2.4270
[train_x, test_x, train_y, test_y] = train_test_generate('cal_phase');
[accuracy, error_dis] = train_and_test(train_x, test_x, train_y, test_y, 'default');

% default
% accuracy = 0.6811, error_dis = 0.4333
[train_x, test_x, train_y, test_y] = train_test_generate('default');
[accuracy, error_dis] = train_and_test(train_x, test_x, train_y, test_y, 'default');

% correlation
% accuracy = 0.0833, error_dis = 2.2241
[train_x, test_x, train_y, test_y] = train_test_generate('default');
[accuracy, error_dis] = train_and_test(train_x, test_x, train_y, test_y, 'correlation');

% with difference
% accuracy = 0.6847, error_dis = 0.4469
[train_x, test_x, train_y, test_y] = train_test_generate('difference');
[accuracy, error_dis] = train_and_test(train_x, test_x, train_y, test_y, 'default');

% with difference + weight
% accuracy = 0.4125, error_dis = 1.2849
[train_x, test_x, train_y, test_y] = train_test_generate('difference');
[accuracy, error_dis] = train_and_test(train_x, test_x, train_y, test_y, 'weight');