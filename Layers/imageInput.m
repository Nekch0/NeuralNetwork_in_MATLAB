classdef imageInput
    properties
        type
        inputSize
        outputSize
    end
    
    methods
        function status = imageInput(inputSize)
            if nargin > 0
                status.type       = 'Input';
                status.inputSize  = cell2mat(inputSize);
                status.outputSize = [1 status.inputSize];
            end
        end
        
        function [output, status] = forward(status, input, batch_size)
            % Check inputSize
            inputSize_ = size(input);
            inputDim_  = ndims(input);
            inputNum_  = status.outputSize(1) * batch_size;
            if inputDim_ == 3
                inputSize_ = [inputSize_, 1];
            end
            if (inputSize_(1) ~= inputNum_) || ~all(inputSize_(2:4) == status.outputSize(2:4))
                error('Error: inputSize is %s but %s\n', mat2str([inputNum_ status.outputSize(2:4)]), mat2str(inputSize_));
            end
            
            output = input;
            
        end
    end
end