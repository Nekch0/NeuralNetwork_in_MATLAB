classdef Model
    properties(SetAccess=private)
        layers
        optimizer
        loss_type
        batch_size 
        iteration
        learning_rate
        accuracy_seq
        loss_seq
        val_accuracy_seq
        val_loss_seq
        dashboard
        axes_accuracy
        plot_accuracy_train
        plot_accuracy_test
        axes_loss
        plot_loss_train
        plot_loss_test
    end
    
    methods
        
        function model = Model(layers, optimizer, loss_type)
            model.layers    = layers;
            model.optimizer = optimizer;
            model.loss_type = loss_type;
            model.accuracy_seq = [];
            model.loss_seq = [];
            model.val_accuracy_seq = [];
            model.val_loss_seq = [];
        end

        function model = compile(model)
            for i = 1:length(model.layers)
                if i > 1
                    model.layers{i} = model.layers{i}.set_IOSize(inputSize);
                end
                inputSize = model.layers{i}.outputSize;
            end
            
            disp("===== Model Summary ===========================");
            disp(" ")
            for i = 1:length(model.layers)
                disp(model.layers{i});
            end
            disp("===============================================");
        end

        function [output, model] = prediction(model, input, batch_size, eval_type)
            for i = 1:length(model.layers)
                if isequal(model.layers{i}.type, 'BatchNorm1D') || isequal(model.layers{i}.type, 'BatchNorm2D')
                    [input, model.layers{i}] = model.layers{i}.forward(input, batch_size, eval_type);
                else
                    [input, model.layers{i}] = model.layers{i}.forward(input, batch_size);
                end
            end
            output = input;
        end
        
        function model = evaluate(model, datas, labels, eval_type, print)
            if nargin < 4
                eval_type = '__evaluate__';
            end
            if nargin < 5
                print = true;
            end
            batch_size_ = model.batch_size * 4;
            data_size   = size(datas);
            data_num    = data_size(1);
            iteration_  = max(floor(data_num / batch_size_), 1);
            rest_size   = mod(data_num, batch_size_);
            if rest_size > 0
                iteration_ = iteration_ + 1;
            end
            lebel_size = size(labels);
            output = zeros(lebel_size(1), lebel_size(end));
            estimated_labels = zeros(lebel_size(1), lebel_size(end));

            % Forward
            for itr = 1:iteration_
                batch_ = batch_size_;
                head = 1 + (itr - 1) * batch_;
                if itr == iteration_ && rest_size > 0
                    tail = data_num;
                    batch_ = rest_size;
                else
                    tail = itr * batch_;
                end

                [output(head:tail, :), ~] = model.prediction(datas(head:tail, :, :), batch_, 'test');
                [~, argmax] = max(output(head:tail, :), [], 2);
                    
                for i = 1:batch_
                    estimated_labels(head+i-1, argmax(i)) = 1;
                end
            end
            
            % Evaluate
            if isequal(eval_type, '__test__') || isequal(eval_type, '__train__')
                model = model.loss_function(labels, output, eval_type, print);
                model = model.accuracy(labels, estimated_labels, eval_type, print);
            else
                model.loss_function(labels, output, eval_type, print);
                model.accuracy(labels, estimated_labels, eval_type, print);  
            end
            if (isequal(eval_type, '__test__') || isequal(eval_type, '__evaluate__')) && print == true
                fprintf('\n');
            end
        end
        
        function model = accuracy(model, test_id, estimeted_id, eval_type, print)
            if nargin < 5
                print = true;
            end
            check = all(test_id == estimeted_id, 2);
            accuracy = (sum(check, 1) / length(check) * 100);
            if isequal(eval_type, '__test__')
                if print == true
                    fprintf("Val Acc: %.4f%% | ", accuracy);
                end
                model.val_accuracy_seq = [model.val_accuracy_seq accuracy];
            else
                if print == true
                    fprintf("Acc: %.4f%% | ", accuracy);
                end
                model.accuracy_seq = [model.accuracy_seq accuracy];
            end
        end
        
        function model = loss_function(model, t, y, eval_type, print)
            if nargin < 5
                print = true;
            end
            if isequal(model.loss_type, 'cross-entropy')
                loss = -sum(t .* log(y + 1e-20), 2);
            else
                loss = sum((y - t) .* (y - t), 2);
            end
            loss = abs(mean(loss));
            if isequal(eval_type, '__test__')
                if print == true
                    fprintf('Val Loss: %.4f | ', loss);
                end
                model.val_loss_seq = [model.val_loss_seq loss];
            else
                if print == true
                    fprintf('Loss: %.4f | ', loss);
                end
                model.loss_seq = [model.loss_seq loss];
            end
        end
          
        function loss = loss_function_back(model, t, y, batch_size)
            if isequal(model.loss_type, 'cross-entropy')
                loss = y - t;
            else
                loss = 2 * (y - t);
            end
            loss = loss ./ batch_size;
        end
        
        function model = fit(model, train_datas, train_labels, learning_rate, epoch, batch_size, test_datas, test_labels)
            data_size = size(train_datas);
            data_num  = data_size(1);
            model.learning_rate = learning_rate;
            model.batch_size = batch_size;
            model.iteration  = max(floor(data_num / model.batch_size), 1);
            rest_size = mod(data_num, model.batch_size);
            if rest_size > 0
                model.iteration = model.iteration + 1;
            end

            model.accuracy_seq = [];
            model.loss_seq = [];
            model.val_accuracy_seq = [];
            model.val_loss_seq = [];
            
            fprintf('\n');
            fprintf('   Learning rate: %g\n', model.learning_rate);
            fprintf('      Batch Size: %d\n', model.batch_size);
            fprintf('     Total Epoch: %d\n', epoch);
            fprintf(' Total Iteration: %d\n', model.iteration);
            fprintf('\n');
            
            % GUI Plot
            Dashboard;
            model.dashboard = findall(0, 'Tag', 'Dashboard');
            model = model.init_plot();
            model = model.evaluate(train_datas, train_labels, '__train__', false);
            model = model.evaluate(test_datas , test_labels , '__test__' , false);
            model = model.update_plot();

            % Fit
            for epc = 1:epoch 
                fprintf('Epoch: %d - ', epc);
                
                % Random Selection from Training Data
                random_indices        = randperm(data_num);
                train_datas_shuffled  = train_datas(random_indices, :, :);
                train_labels_shuffled = train_labels(random_indices, :);
                
                % Training
                for itr = 1:model.iteration
                    if mod(itr, floor(model.iteration/10)) == 0
                        fprintf('*');
                    end
                    batch_ = model.batch_size;
                    head = 1 + (itr - 1) * model.batch_size;
                    if itr == model.iteration && rest_size > 0
                        tail = data_num;
                        batch_ = rest_size;
                    else
                        tail = itr * model.batch_size;
                    end
                    input = train_datas_shuffled(head : tail, :, :);
                    label = train_labels_shuffled(head : tail, :);
                    [out, model] = model.prediction(input, batch_, 'train');      % Prediction
                    loss_grad    = model.loss_function_back(label, out, batch_);  % Calculate Loss
                    model        = model.backpropagate(loss_grad, batch_, itr);   % Back Propagation
                end
                
                fprintf(' | ');
                model = model.evaluate(train_datas, train_labels, '__train__');
                model = model.evaluate(test_datas , test_labels , '__test__');
                model = model.update_plot();
            end
            
            fprintf('TrainData : '); model.evaluate(train_datas, train_labels);
            fprintf(' TestData : '); model.evaluate(test_datas, test_labels);
        end
        
        function model = backpropagate(model, loss_grad, batch_size, itration)
            
            % Call Backward Function
            for i = length(model.layers):-1:1
                if ~isequal(model.layers{i}.type, 'Input')
                    [loss_grad, model.layers{i}] = model.layers{i}.backward(loss_grad, batch_size);
                end
            end
            
            % Call Update Function
            for i = length(model.layers):-1:1
                if ~isequal(model.layers{i}.type, 'Input')
                    model.layers{i} = model.layers{i}.update(model.optimizer, model.learning_rate, itration);
                end
            end
            
        end
        
        function model = init_plot(model)
            if ~isempty(model.dashboard)
                model.axes_accuracy = findobj(model.dashboard, 'Tag', 'plot_accuracy');
                model.axes_loss     = findobj(model.dashboard, 'Tag', 'plot_loss');
                xlabel(model.axes_accuracy, 'Epoch'   , 'Interpreter', 'latex', 'FontSize', 20, 'Color', '#F9FAF8'); 
                ylabel(model.axes_accuracy, 'Accuracy', 'Interpreter', 'latex', 'FontSize', 20, 'Color', '#F9FAF8');
                xlabel(model.axes_loss    , 'Epoch'   , 'Interpreter', 'latex', 'FontSize', 20, 'Color', '#F9FAF8'); 
                ylabel(model.axes_loss    , 'Loss'    , 'Interpreter', 'latex', 'FontSize', 20, 'Color', '#F9FAF8');
				xlim(model.axes_accuracy, [0, 1]);
                xlim(model.axes_loss    , [0, 1]);
                set(model.axes_accuracy, 'XTick', 0:1);
                set(model.axes_loss    , 'XTick', 0:1);
                set(model.axes_accuracy, 'XColor', '#F9FAF8');
                set(model.axes_accuracy, 'YColor', '#F9FAF8');
                set(model.axes_loss    , 'XColor', '#F9FAF8');
                set(model.axes_loss    , 'YColor', '#F9FAF8');
                set(model.axes_accuracy, 'Color', '#EAEAE9');
                set(model.axes_loss    , 'Color', '#EAEAE9');
                grid(model.axes_accuracy, 'on');
                grid(model.axes_loss    , 'on');
                set(model.axes_accuracy, 'GridColor', '#1C2638');
                set(model.axes_loss    , 'GridColor', '#1C2638');
                drawnow;
            end
        end
        
        function model = update_plot(model)
            len = length(model.accuracy_seq)-1;
            epoch = 0:len;
            
            if ~isempty(findall(0, 'Tag', 'Dashboard'))
                hold(model.axes_accuracy, 'on');
                if(len == 0)
                    model.plot_accuracy_train = plot(model.axes_accuracy, epoch, model.accuracy_seq    , 'o-', 'Color', '#46D369', 'LineWidth', 2, 'DisplayName', 'Train');
                    model.plot_accuracy_test  = plot(model.axes_accuracy, epoch, model.val_accuracy_seq, 'o-', 'Color', '#E87E27', 'LineWidth', 2, 'DisplayName', 'Test');
                else
                    set(model.plot_accuracy_train, 'XData', epoch, 'YData', model.accuracy_seq);
                    set(model.plot_accuracy_test , 'XData', epoch, 'YData', model.val_accuracy_seq);
                end
                model.plot_accuracy_train.MarkerFaceColor = '#46D369';
                model.plot_accuracy_train.MarkerSize = 4;
                model.plot_accuracy_test.MarkerFaceColor = '#E87E27';
                model.plot_accuracy_test.MarkerSize = 4;
                hold(model.axes_accuracy, 'off');
            
                hold(model.axes_loss, 'on');
                if(len == 0)
                     model.plot_loss_train = plot(model.axes_loss, epoch, model.loss_seq    , 'o-', 'Color', '#46D369', 'LineWidth', 2, 'DisplayName', 'Train');
                     model.plot_loss_test  = plot(model.axes_loss, epoch, model.val_loss_seq, 'o-', 'Color', '#E87E27', 'LineWidth', 2, 'DisplayName', 'Test');
                else
                    set(model.plot_loss_train, 'XData', epoch, 'YData', model.loss_seq);
                    set(model.plot_loss_test , 'XData', epoch, 'YData', model.val_loss_seq);
                end
                model.plot_loss_train.MarkerFaceColor = '#46D369';
                model.plot_loss_train.MarkerSize = 4;
                model.plot_loss_test.MarkerFaceColor = '#E87E27';
                model.plot_loss_test.MarkerSize = 4;
                hold(model.axes_loss, 'off');
                
                legend(model.axes_accuracy,'Location','southeast', 'Color', 'none', 'Box', 'off');
                legend(model.axes_loss, 'Color', 'none', 'Box', 'off');
                xlabel(model.axes_accuracy, 'Epoch'   , 'Interpreter', 'latex', 'FontSize', 20, 'Color', '#F9FAF8'); 
                ylabel(model.axes_accuracy, 'Accuracy', 'Interpreter', 'latex', 'FontSize', 20, 'Color', '#F9FAF8');
                xlabel(model.axes_loss    , 'Epoch'   , 'Interpreter', 'latex', 'FontSize', 20, 'Color', '#F9FAF8'); 
                ylabel(model.axes_loss    , 'Loss'    , 'Interpreter', 'latex', 'FontSize', 20, 'Color', '#F9FAF8');
                xlim(model.axes_accuracy, [0, max(1, len)]);
                xlim(model.axes_loss    , [0, max(1, len)]);
                if length(model.val_accuracy_seq) >= 2
                    ylim(model.axes_accuracy, [min(mean(model.val_accuracy_seq), model.val_accuracy_seq(2)), 100]);
                    ylim(model.axes_loss    , [0, max(mean(model.val_loss_seq), model.val_loss_seq(2))]);
                else
                    ylim(model.axes_accuracy, [mean(model.val_accuracy_seq), 100]);
                    ylim(model.axes_loss    , [0, mean(model.val_loss_seq)]);
                end
                if max(2, len) < 10
                    set(model.axes_accuracy, 'XTick', 0:max(1, len));
                    set(model.axes_loss    , 'XTick', 0:max(1, len));
                    set(model.axes_accuracy, 'XTicklabel', 0:max(1, len));
                    set(model.axes_loss    , 'XTicklabel', 0:max(1, len));
                elseif len == 10
                    set(model.axes_accuracy, 'XTickMode', 'auto');
                    set(model.axes_loss    , 'XTickMode', 'auto');
                    set(model.axes_accuracy, 'XTicklabelMode', 'auto');
                    set(model.axes_loss    , 'XTicklabelMode', 'auto');
                end
                grid(model.axes_accuracy, 'on');
                grid(model.axes_loss    , 'on');
                set(model.axes_accuracy, 'GridColor', '#1C2638');
                set(model.axes_loss    , 'GridColor', '#1C2638');
                drawnow;
            end
        end
        
    end  
end