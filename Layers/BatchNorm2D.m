classdef BatchNorm2D
    properties
        type
        inputSize
        outputSize
        channels
        gamma
        gamma_grad
        gamma_buffer
        beta
        beta_grad
        beta_buffer
        momentum
        input_dev
        input_var
        input_std
        running_var
        running_mean
        data_buffer_layer
        beta1
        beta2
        mg
        vg
        mb
        vb
    end
    
    methods
        function status = BatchNorm2D(momentum)
            status.type = 'BatchNorm2D';
            status.momentum = momentum;
            status.beta1 = 0.9;
            status.beta2 = 0.999;
        end
        
        function status = set_IOSize(status, inputSize)
            if iscell(inputSize)
                inputSize = cell2mat(inputSize);
            end
            status.inputSize  = inputSize;
            status.outputSize = inputSize;
            status.channels   = inputSize(end);
            status.gamma = ones([1 1 1 inputSize(end)]);
            status.beta  = zeros([1 1 1 inputSize(end)]);
            status.gamma_buffer = zeros(size(status.gamma));
            status.beta_buffer  = zeros(size(status.beta));
            status.running_var  = zeros([1 inputSize(2:end)]);
            status.running_mean = zeros([1 inputSize(2:end)]);            
            status.mg = zeros(size(status.gamma));
            status.vg = zeros(size(status.gamma));
            status.mb = zeros(size(status.beta));
            status.vb = zeros(size(status.beta));
        end
        
        function [output, status] = forward(status, input, batch_size, eval_type)
            
            % Check InputSize
            inputSize_ = size(input);
            inputDim_  = ndims(input);
            inputNum_  = status.outputSize(1) * batch_size;
            if inputDim_ == 3
                inputSize_ = [inputSize_, 1];
            end
            if (inputSize_(1) ~= inputNum_) || ~all(inputSize_(2:4) == status.inputSize(2:4))
                error('Error: inputSize is %s but %s\n', mat2str([inputNum_ status.inputSize(2:4)]), mat2str(inputSize_));
            end
            
            if isequal(eval_type, 'train')
                input_mean = mean(input, [1, 2, 3]);
                status.input_dev = input(:, :, :, :) - input_mean(:, :, :, :);
                status.input_var = var(input, 0, [1, 2, 3]);
                status.input_std = sqrt(status.input_var + 1e-7);
                input_hat  = status.input_dev ./ status.input_std;
                status.running_mean = status.momentum * status.running_mean + (1 - status.momentum) * input_mean;
                status.running_var  = status.momentum * status.running_var  + (1 - status.momentum) * status.input_var;
            else
                dev = input - status.running_mean;
                input_hat = dev ./ sqrt(status.running_var + 1e-7);
            end
            output = status.gamma .* input_hat + status.beta;
            status.data_buffer_layer = output;
        end
        
        function [output, status] = backward(status, input, ~)
            status.beta_grad  = sum(input, [1, 2, 3]);     
            status.gamma_grad = sum(input .* status.data_buffer_layer, [1 2 3]); 

            inputSize_ = size(input);
            batch_ = prod(inputSize_(1:3));
            input_hat_grad = input .* status.gamma;
            
            var_grad  = -sum(input_hat_grad .* status.input_dev, [1 2 3]) ./ (2 * status.input_std.^3);
            mean_grad = -sum(input_hat_grad, [1 2 3]) ./ status.input_std + var_grad .* (-2) .* sum(status.input_dev, [1 2 3]) / batch_;
            output = input_hat_grad ./ status.input_std + var_grad .* 2 .* status.input_dev / batch_ + mean_grad / batch_;
        end
        
        function status = update(status, optimizer, learning_rate, iteration)
            db = 0;
            dg = 0;
            if isequal(optimizer, 'Adagrad')
                status.beta_buffer  = status.beta_buffer  + status.beta_grad  .* status.beta_grad;
                status.gamma_buffer = status.gamma_buffer + status.gamma_grad .* status.gamma_grad;
                db = learning_rate .* status.beta_grad ./ (sqrt(status.beta_buffer) + 1e-7);
                dg = learning_rate .* status.beta_grad ./ (sqrt(status.gamma_buffer) + 1e-7);
            elseif isequal(optimizer, 'Adam')                
                alpha = learning_rate * sqrt(1.0 - status.beta2^iteration) / (1.0 - status.beta1^iteration);
                status.mb = status.mb + (status.beta_grad     - status.mb) * (1 - status.beta1);
                status.vb = status.vb + (status.beta_grad.^2  - status.vb) * (1 - status.beta2);
                status.mg = status.mg + (status.gamma_grad    - status.mg) * (1 - status.beta1);
                status.vg = status.vg + (status.gamma_grad.^2 - status.vg) * (1 - status.beta2);
                db = alpha .* status.mb ./ (sqrt(status.vb) + 1e-8);
                dg = alpha .* status.mg ./ (sqrt(status.vg) + 1e-8);
            end
            status.beta  = status.beta - db;
            status.gamma = status.gamma - dg;
        end

    end
end