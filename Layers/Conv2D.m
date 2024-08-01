classdef Conv2D
    properties
        type
        inputChannel
        outputChannel
        filterSize
        stride
        activation_type
        filter
        filter_grad
        filter_buffer
        bias
        bias_grad
        bias_buffer
        padding_type
        padding
        inputSize
        outputSize
        data_buffer_layer
        data_buffer_activation
        beta1
        beta2
        mf
        vf
        mb
        vb
    end
    
    methods
        function status = Conv2D(outputChannel, filterSize, stride, padding_type, activation)
            if nargin > 0
                status.type            = 'Conv2D';
                status.outputChannel   = outputChannel;
                status.filterSize      = cell2mat(filterSize);
                status.stride          = stride;
                status.padding_type    = padding_type;
                if nargin < 5
                    status.activation_type = 'None';
                else
                    status.activation_type = activation;
                end
                status.beta1 = 0.9;
                status.beta2 = 0.999;
            end
        end
        
        function status = set_IOSize(status, inputSize)
            if iscell(inputSize)
                inputSize = cell2mat(inputSize);
            end
            status.inputSize    = inputSize;
            status.inputChannel = inputSize(4);
            if isequal(status.padding_type, 'SAME')
                    status.padding = floor(((status.inputSize(1:2) - 1) * status.stride + status.filterSize - status.inputSize(1:2)) / 2);
            else
                    status.padding = 0;
            end
            status.outputSize = floor((status.inputSize(2:3) - status.filterSize + 2 * status.padding) / status.stride) + 1;
            status.outputSize = [status.inputSize(1) status.outputSize, status.outputChannel];
            status.filter        = randn([status.filterSize status.inputSize(end) status.outputChannel]) * sqrt(2.0 / prod([status.filterSize status.inputSize(3) status.outputChannel]));
            status.bias          = zeros([1 status.outputChannel]);
            status.filter_buffer = zeros(size(status.filter));
            status.bias_buffer   = zeros(size(status.bias));
            status.mf = zeros(size(status.filter));
            status.vf = zeros(size(status.filter));
            status.mb = zeros(size(status.bias));
            status.vb = zeros(size(status.bias));
        end

        function [featureMap, status] = activation(status, input)
            status.data_buffer_activation = input;
            if isequal(status.activation_type, 'ReLU')
                featureMap = max(input, 0);
            else
                featureMap = input;
            end
        end
        
        function grad = activation_back(status, input)
            if isequal(status.activation_type, 'ReLU')
                grad = (status.data_buffer_activation > 0) .* input;
            elseif isequal(status.activation_type, 'SoftMax')
                grad = input;
            else
                grad = input;
            end
        end
        
        function output = img2col(status, input, N, filterSize_, outputSize_, inputChannel_)
            % N x FH x FW x OH x OW x iC
            output = zeros([N, filterSize_, outputSize_, inputChannel_]);
            H = outputSize_(1) - 1;
            W = outputSize_(2) - 1;
            for i = 1:filterSize_(1)
            	for j = 1:filterSize_(2)
                    output(:, i, j, :, :, :) = input(:, i:status.stride:i+status.stride*H, j:status.stride:j+status.stride*W, :);    
            	end
            end
            output = permute(output, [1 4 5 6 2 3]);
            output = reshape(output, [prod([N outputSize_]) prod([filterSize_ inputChannel_])]);
        end

        function output = reshape_filter(~, filter_, shape_)
            output = reshape(filter_, shape_);
        end
        
        function [featureMap, status] = forward(status, input, batch_size)
            
            % Check inputSize
            inputSize_ = size(input);
            inputDim_  = ndims(input);
            inputNum_  = status.outputSize(1) * batch_size;
            if inputDim_ == 3
                inputSize_ = [inputSize_, 1];
            end
            if (inputSize_(1) ~= inputNum_) || ~all(inputSize_(2:4) == status.inputSize(2:4))
                error('Error: inputSize is %s but %s\n', mat2str([inputNum_ status.inputSize(2:4)]), mat2str(inputSize_));
            end

            status.data_buffer_layer = input;
            input = padarray(input, [0 status.padding status.padding 0]);
            %fprintf("\nInput:");disp(size(input));
            %disp([inputNum_, status.filterSize, status.outputSize(2:3), status.inputChannel]);

            input_ = status.img2col(input, inputNum_, status.filterSize, status.outputSize(2:3), status.inputChannel);
            % Batch x  H x  W x iC -> Batch*OH*OW x FH*FW*iC
            %    47 x 16 x 16 x  1 ->    47*14*14 x  3* 3* 1  ->   9212 x 9
            %fprintf('Reshaped I:'); disp(size(input));

            filter_ = status.reshape_filter(status.filter, [prod([status.filterSize status.inputSize(4)]) status.outputChannel]);
            % FH x FW x iC x oC -> FH*FW*iC x oC
            %  3 x  3 x  1 x 16 ->  3* 3* 1 x 16  ->  9 x 16
            %fprintf('Reshaped F:'); disp(size(filter_));
            
            % 入力画像全体(n枚)に対して１枚のフィルタを用いて畳み込みを行い１枚の出力を得る
            % (Batch*OH*OW x FH*FW*iC) x (FH*FW*iC x oC) = Batch*OH*OW x oC
            % -> (9212 x 9) x (9 x 16) = 9212 x 16
            %fprintf('featureMap:'); disp(size(featureMap));
            %disp(input(1, :, :, 1));
            featureMap = input_ * filter_ + status.bias;
            featureMap = reshape(featureMap, [inputNum_ status.outputChannel status.outputSize(2:3)]);
            featureMap = permute(featureMap, [1 3 4 2]);
            %disp(featureMap(1, :, :, 1));
            
            % 畳み込み演算
            if false
                featureMap_ = zeros([inputNum_ status.outputSize(2:4)]);    
                for n = 1:inputNum_
                    for i = 1 : status.stride : status.inputSize(2) - status.filterSize(1) + 1
                        for j = 1 : status.stride : status.inputSize(3) - status.filterSize(2) + 1
                            region = squeeze(input(n, i : i + status.filterSize - 1, j : j + status.filterSize - 1, :));
                            %disp(size(region));
                            %disp(size(status.filter));
                            hadamard = region .* status.filter;
                            %disp(size(hadamard));
                            feature = transpose(squeeze(sum(hadamard, [1, 2, 3])));
                            feature = feature + status.bias;
                            featureMap_(n, (i-1)/status.stride+1, (j-1)/status.stride+1, :) = feature;
                        
                        end
                    end
                end
                disp(featureMap_(1, :, :, 1));a
            end

            
            [featureMap, status] = status.activation(featureMap);
        end
        
        function [grad, status] = backward(status, input, batch_size)
            input = status.activation_back(input); 
            
            inputSize_ = size(input);
            inputNum_  = status.inputSize(1) * batch_size;
            
            % Pad inputGrad
            innerpadSize = [inputNum_ status.outputSize(2:3) + status.stride .* (status.outputSize(2:3) - 1), status.outputChannel];
            input_padded = zeros(innerpadSize);
            for i = 1:status.outputSize(3)
                for j = 1:status.outputSize(3)
                    input_padded(:, 1 + (i-1) * (status.stride + 1), 1 + (j-1) * (status.stride + 1), :) = input(:, i, j, :);
                end
            end
            input_padded = padarray(input_padded, [0 status.padding status.padding 0]);
            input_col    = status.img2col(input_padded, inputNum_, status.filterSize, status.inputSize(2:3), status.outputChannel);
            % Batch x  H x  W x iC -> Batch*OH*OW x FH*FW*iC
            %    47 x  5 x  5 x 32 ->    47* 7* 7 x  3* 3*32  ->   2303 x 288 
            %fprintf('Reshaped I:'); disp(size(input_padded));
            
            % Reshape Filter
            %rotated_filter = zeros(size(status.filter));
            %for i = 1:status.outputChannel
            %    for j = 1:status.inputChannel
            %        rotated_filter(:, :, j, i) = rot90(status.filter(:, :, j, i), 2);
            %    end
            %end
            %filter_ = permute(rotated_filter, [1, 2, 4, 3]);
            
            filter_ = permute(status.filter, [1, 2, 4, 3]);
            filter_ = status.reshape_filter(filter_, [prod([status.filterSize status.outputSize(4)]) status.inputChannel]);
            % FH x FW x iC x oC -> FH*FW*iC x oC
            %  3 x  3 x 32 x 16 ->  3* 3*32 x 16  -> 288 x 16
            %fprintf('Reshaped F:'); disp(size(filter_));
            grad = input_col * filter_;
            grad = reshape(grad, [inputNum_ status.inputChannel status.inputSize(2:3)]);
            grad = permute(grad, [1 3 4 2]);
            % 入力画像全体(n枚)に対して１枚のフィルタを用いて畳み込みを行い１枚の出力を得る
            % (Batch*OH*OW x FH*FW*iC) x (FH*FW*iC x oC) = Batch*OH*OW x oC
            % -> (2303 x 288) x (288 x 16) = 2303 x 16
            %fprintf('featureMap:'); disp(size(grad));

            input_buffer = status.img2col(status.data_buffer_layer, inputNum_, status.filterSize, status.outputSize(2:3), status.inputChannel);
            input_       = status.reshape_filter(input, [prod(inputSize_(1:3)) status.outputChannel]);
            status.filter_grad = input_buffer' * input_;
            status.filter_grad = reshape(status.filter_grad, size(status.filter));

            input_sum = transpose(squeeze(sum(input, [1 2 3])));
            status.bias_grad = input_sum;
        end
        
        function status = update(status, optimizer, learning_rate, iteration)
            df = 0;
            db = 0;
            if isequal(optimizer, 'Adagrad')
                status.filter_buffer = status.filter_buffer + status.filter_grad .* status.filter_grad;
                status.bias_buffer   = status.bias_buffer + status.bias_grad .* status.bias_grad;
                df = learning_rate .* status.filter_grad ./ (sqrt(status.filter_buffer) + 1e-7);
                db = learning_rate .* status.bias_grad   ./ (sqrt(status.bias_buffer) + 1e-7);
            elseif isequal(optimizer, 'Adam')                    
                alpha = learning_rate * sqrt(1.0 - status.beta2^iteration) / (1.0 - status.beta1^iteration);
                status.mf = status.mf + (status.filter_grad    - status.mf) * (1 - status.beta1);
                status.vf = status.vf + (status.filter_grad.^2 - status.vf) * (1 - status.beta2);
                status.mb = status.mb + (status.bias_grad      - status.mb) * (1 - status.beta1);
                status.vb = status.vb + (status.bias_grad.^2   - status.vb) * (1 - status.beta2);
                df = alpha .* status.mf ./ (sqrt(status.vf) + 1e-8);
                db = alpha .* status.mb ./ (sqrt(status.vb) + 1e-8);
            end
            status.filter = status.filter - df;     %disp(df(:, :, 1, 1)); disp(status.filter_grad(:, :, 1, 1));
            status.bias   = status.bias   - db;     %disp(db(1, 1:5));
        end
        
    end
end