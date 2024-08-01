function [ train_data , train_id , test_data , test_id ] = loadUSPS()
    train_data =[]; train_id =[]; test_data =[]; test_id =[];
    try
        load './Dataset/USPS.mat';
    catch
        disp ('Error : the file usps_resampled . mat was not found.')
        return
    end
    train_data_size = 7292;
	test_data_size  = 2007;
    if test_data_size > 2007
        disp ('Error : number of test data was too large.')
        return ;
    end
    train_data = fea(1: train_data_size ,:);
    train_id   = gnd(1: train_data_size );
    test_data  = fea(7292:7292 + test_data_size -1 ,:);
    test_id    = gnd(7292:7292 + test_data_size -1);
end