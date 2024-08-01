classdef Dense
    properties
        type
        neuronNum
        weight
        weight_grad
        weight_buffer
        bias
        bias_grad
        bias_buffer
        activation_type
        inputSize
        outputSize
        data_buffer_layer
        data_buffer_activation
        beta1
        beta2
        mw
        vw
        mb
        vb
    end
    
    methods
        function status = Dense(neuronNum, activation)
            if nargin > 0
                status.type      = 'Dense';
                status.neuronNum = neuronNum;
                if nargin < 2
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
            status.inputSize     = inputSize;
            status.outputSize    = [inputSize(1) status.neuronNum];
            status.bias          = zeros([1 status.neuronNum]);
            status.bias_buffer   = zeros([1 status.neuronNum]);
            status.weight        = rand([inputSize(2), status.neuronNum]) * sqrt(2.0 / status.neuronNum);
            status.weight_buffer = zeros([inputSize(2), status.neuronNum]);
            status.mw = zeros(size(status.weight));
            status.vw = zeros(size(status.weight));
            status.mb = zeros(size(status.bias));
            status.vb = zeros(size(status.bias));
        end
        
        function [feature, status] = activation(status, input)
            status.data_buffer_activation = input;
            if isequal(status.activation_type, 'ReLU')
                feature = max(input, 0);
            elseif isequal(status.activation_type, 'SoftMax')
                input = transpose(input);
                C = -max(input);
                feature = exp(input + C) ./ sum(exp(input + C), 1);
                feature = transpose(feature);
            else
                feature = input;
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
        
        function [feature, status] = forward(status, input, batch_size)
            % Check inputSize
            inputNum_  = status.inputSize(1) * batch_size;
            inputSize_ = size(input);
            if (inputNum_ ~= inputSize_(1)) || ~all(inputSize_(2) == status.inputSize(2))
                error('Error: inputSize is %s but %s\n', mat2str([inputNum_ status.inputSize(2)]), mat2str(inputSize_));
            end
            
            status.data_buffer_layer = input;
            feature = input * status.weight + status.bias; 
            [feature, status] = status.activation(feature);  
        end
        
        function [grad, status] = backward(status, input, ~)
            % 　　入力の勾配 : dL/dX = dL/dY * W^T : (Batch, CurD)*(CurD, PreD)  -> (Batch CurD) 
            % 　　重みの勾配 : dL/dW = X^T * dL/dY : (PreD, Batch)*(Batch, CurD) -> (PreD CurD) 
            % バイアスの勾配 : dL/Db = Σ^N dL/dy   : (Batch, CurD)               -> (1 CurD) 
            
            input_grad = status.activation_back(input);
            grad       = input_grad * status.weight';   
            
            status.weight_grad = status.data_buffer_layer' * input;
            status.bias_grad   = sum(input, 1);
        end
        
        function status = update(status, optimizer, learning_rate, iteration) 
            dw = 0;
            db = 0;
            if isequal(optimizer, 'Adagrad')
                status.weight_buffer = status.weight_buffer + status.weight_grad.^2;
                status.bias_buffer   = status.bias_buffer   + status.bias_grad.^2;
                dw = learning_rate .* status.weight_grad ./ (sqrt(status.weight_buffer) + 1e-7);
                db = learning_rate .* status.bias_grad   ./ (sqrt(status.bias_buffer) + 1e-7);
            elseif isequal(optimizer, 'Adam')                
                alpha = learning_rate * sqrt(1.0 - status.beta2^iteration) / (1.0 - status.beta1^iteration);
                status.mw = status.mw + (status.weight_grad    - status.mw) * (1 - status.beta1);
                status.vw = status.vw + (status.weight_grad.^2 - status.vw) * (1 - status.beta2);
                status.mb = status.mb + (status.bias_grad      - status.mb) * (1 - status.beta1);
                status.vb = status.vb + (status.bias_grad.^2   - status.vb) * (1 - status.beta2);
                dw = alpha .* status.mw ./ (sqrt(status.vw) + 1e-8);
                db = alpha .* status.mb ./ (sqrt(status.vb) + 1e-8);
            end
            status.weight = status.weight - dw;
            status.bias   = status.bias - db;
        end
        
    end
end