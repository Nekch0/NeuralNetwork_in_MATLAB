classdef Pool
    properties
        type
        channels
        poolingType
        poolingSize
        stride
        inputSize
        outputSize
        data_buffer_layer
    end
    
    methods
        function status = Pool(poolingType, poolingSize, stride)
            if nargin > 0
                status.type        = 'Pool';
                status.poolingType = poolingType;
                status.poolingSize = cell2mat(poolingSize);
                status.stride      = stride;
            end
        end
        
        function status = set_IOSize(status, inputSize)
            if iscell(inputSize)
                inputSize = cell2mat(inputSize);
            end
            status.inputSize  = inputSize;
            status.outputSize = floor((status.inputSize(2:3) - status.poolingSize) / status.stride) + 1;
            status.outputSize = [status.inputSize(1) status.outputSize status.inputSize(4)];
            status.channels   = status.inputSize(4);
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
            
            % Pooling
            featureMap = zeros([inputNum_ status.outputSize(2:4)]);
            for i = 1 : status.stride : status.inputSize(2) - status.poolingSize + 1
                for j = 1 : status.stride : status.inputSize(3) - status.poolingSize + 1
                    region = input(:, i : i + status.poolingSize - 1, j : j + status.poolingSize - 1, :);
                    if isequal(status.poolingType, 'max')
                        feature = max(region, [], [2, 3]);
                    else
                        feature = mean(region, [2, 3]);
                    end
                    featureMap(:, (i-1)/status.stride+1, (j-1)/status.stride+1, :) = feature;
                end
            end
        end
        
        function [grad, status] = backward(status, input, batch_size)
            grad = zeros(size(status.data_buffer_layer));
            for i = 1 : status.stride : status.inputSize(2) - status.poolingSize + 1
                for j = 1 : status.stride : status.inputSize(3) - status.poolingSize + 1
                    region = status.data_buffer_layer(:, i : i + status.poolingSize - 1, j : j + status.poolingSize - 1, :);
                    if isequal(status.poolingType, 'max')
                        for k = 1 : status.channels
                            for n = 1 : status.inputSize(1) * batch_size
                                [~, argmax] = max(region(n, :, :, k));
                                grad(n, i+argmax(1)-1, j+argmax(2)-1, k) = input(n, floor((i-1)/status.stride)+1, floor((j-1)/status.stride)+1, k);
                            end
                        end
                    else
                        for k = 1 : status.channels
                            for n = 1 : status.inputSize(4) * batch_size
                                grad_mean = input(n, floor((i-1)/status.stride)+1, floor((j-1)/status.stride)+1, k);
                                grad(n, i:i+status.stride-1, j:j+status.stride-1, k) = grad_mean;
                            end
                        end
                    end
                end
            end
        end
        
        function status = update(status, ~, ~, ~)
        end

    end
end