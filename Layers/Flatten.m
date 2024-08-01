classdef Flatten
    properties
        type
        inputSize
        outputSize
    end
    
    methods
        function status = Flatten()
            status.type      = 'Flatten';
        end
        
        function status = set_IOSize(status, inputSize)
            if iscell(inputSize)
                inputSize = cell2mat(inputSize);
            end
            status.inputSize  = inputSize;
            status.outputSize = [inputSize(1), prod(inputSize(2:end))];
        end
        
        function [output, status] = forward(status, input, batch_size)
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
            
            output = reshape(input, [inputNum_, prod(inputSize_(2:4))]);
        end
        
        function [output, status] = backward(status, input, batch_size)
            inputNum_ = status.inputSize(1) * batch_size;
            output = reshape(input, [inputNum_ status.inputSize(2:4)]); 
        end
        
        function status = update(status, ~, ~, ~)
        end

    end
end