%% Load Data
[train_data, train_id, test_data, test_id] = loadUSPS;

% Image Parameter
channel = 1;
height  = 16;
width   = 16;
label_n = 10;

% Vector to Image / Normalization
train_images = zeros(length(train_data), height, width, channel);
test_images  = zeros(length(test_data) , height, width, channel);
for i = 1:length(train_data)
    image = reshape(train_data(i, :), [height width channel])';
    min_val = min(image(:));
    max_val = max(image(:));
    train_images(i, :, :, channel) = (image - min_val) / (max_val - min_val);
end
for i = 1:length(test_data)
    image = reshape(test_data(i, :), [height width channel])';
    min_val = min(image(:));
    max_val = max(image(:));
    test_images(i, :, :, channel) = (image - min_val) / (max_val - min_val);
end

% One-Hot Encoding
train_labels = zeros(length(train_id), label_n);
test_labels  = zeros(length(test_id) , label_n);
for i = 1:length(train_data)
    train_labels(i, train_id(i)) = 1;
end
for i = 1:length(test_data)
    test_labels(i, test_id(i)) = 1;
end

fprintf("Preparation Finished. (Train:%d, Test:%d)\n", length(train_images), length(test_images));

%% Learning
addpath('./Layers');

%layers = {
    %imageInput({16, 16, 1});
    %Conv2D(32, {3, 3}, 1, 'VALID', 'None');
    %BatchNorm2D(0.99);
    %Activation('ReLU');
	%Pool('max', {2, 2}, 2);
    %Conv2D(64, {3, 3}, 1, 'VALID', 'None');
    %BatchNorm2D(0.99);
    %Activation('ReLU');
    %Pool('max', {2, 2}, 2);
    %Flatten();
    %Dense(256, 'ReLU');
    %Dense(128, 'ReLU');
    %Dense(10, 'SoftMax')
%};

layers = {
    imageInput({16, 16, 1});
    Flatten();
    Dense(512);
    BatchNorm1D(0.99);
    Activation('ReLU');
    Dense(128);
    BatchNorm1D(0.99);
    Activation('ReLU');
    Dense(10, 'SoftMax')
};

learning_rate = 0.001;
epoch_size    = 50;
batch_size    = 32;
optimizer     = 'Adam';
loss_function = 'cross-entropy';

%  Data : [Data_Num, Height, Weight, Channel]
% Label : [Data_Num, Label_ID]

rng(42); % 条件を揃えるため乱数は固定

model = Model(layers, optimizer , loss_function);
model = model.compile();
model = model.fit(train_images, train_labels, learning_rate, epoch_size, batch_size, test_images, test_labels);
