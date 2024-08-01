classdef Activation
    properties
        type
        inputChannel
        outputChannel
        activation_type
        inputSize
        outputSize
        data_buffer_activation
    end
    
    methods
        function status = Activation(activation)
            if nargin > 0
                status.type            = 'Activation';
                status.activation_type = activation;
            end
        end
        
        function status = set_IOSize(status, inputSize)
            if iscell(inputSize)
                inputSize = cell2mat(inputSize);
            end
            status.inputSize  = inputSize;
            status.outputSize = inputSize;
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
        
        function [output, status] = forward(status, input, ~)
            [output, status] = status.activation(input);
        end
        
        function [grad, status] = backward(status, input, ~)
            grad = status.activation_back(input); 
        end
        
        function status = update(status, ~, ~, ~)
        end
        
    end
end