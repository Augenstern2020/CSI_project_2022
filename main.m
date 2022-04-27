format long;


% 从.dat转成.csv
[FileName,PathName,FilterIndex] = uigetfile('*.dat','MultiSelect','on');
% 不可以选择单个文件
% 匹配所有的.dat文件; 可以选择多个文件; 多文件模式"开"
% 创建标准的对话框并通过交互式操作取得文件名

% 一次性选择多个文件 最外层的这个大循环负责逐个文件处理
for i = 1:length(FileName)% length求出文件数目
    csi_trace = read_bf_file(strcat(PathName,char(FileName(i))));    %读取文件
    
    % 绝对路径
    % [packet_num, Tx, Rx, Nsub]
    
    % eliminate empty cell - new
    for packet_index = 1:length(csi_trace)
        if csi_trace{packet_index}.Ntx ~= 2
            csi_trace{packet_index}=[];
        end
        %筛选发送天线数量，与频率相关，2.4GHz时2×3，5GHz 1×3
    end

    
    % eliminate empty cell
    xx = find(cellfun('isempty', csi_trace));
    %xx
    % 对csi_trace中的每个元胞应用函数isempty
    % find 查索引
    csi_trace(xx) = [];% eliminate empty cell

    
    % Extract CSI information for each packet
    fprintf('Have raw CSI for %d packets\n', length(csi_trace))
    

    % Scaled into linear
    csi = zeros(length(csi_trace),2,3,30);% 创建全零数组
    tempcsi = zeros(2,3,30);% 两个发射天线
    
    
    % 中间变量，存储函数返回值
    timestamp = zeros(1,length(csi_trace));
    temp = [];
    
    % csi_trace是收到的包的个数
    for packet_index = 1:length(csi_trace)
        %for 2.4GHz
        tempcsi(:,:,:) = get_scaled_csi(csi_trace{packet_index});	%获取CSI值
        csi(packet_index,:,:,:) = tempcsi(:,:,:);

        %for 5Ghz
        %csi(packet_index,:,:) = get_scaled_csi(csi_trace{packet_index});
        %处理为1*3*30的数据


        timestamp(packet_index) = csi_trace{packet_index}.timestamp_low * 1.0e-6;


        %用于显示
        %csi_trace{packet_index}
        %timestamp(packet_index)
        %timestamp(packet_index)
        %csi_trace{packet_index}.timestamp_low
    end
    timestamp = timestamp';

    % File export

    % 幅度
    csi_amp_matrix = db(abs(squeeze(csi)));% length*2*3*30
    
    % 相位
    csi_phase_matrix = angle(squeeze(csi));
    csi_phase_matrix2 = zeros(length(csi_trace),2,3,30);
    
    
%     for kk =1:30
%         for jj = 1:3
%             for ii = 1:2
%                 test(ii,jj,kk) = 3*2*(kk-1) + 2*(jj-1)+ii;
%             end
%         end
%     end
    

    for ii=1:size(csi_phase_matrix,1)
        for j=1:size(csi_phase_matrix,2)
            for k = 1:size(csi_phase_matrix,3)
                corrected_phase = phase_calibration(csi_phase_matrix(ii,j,k,:)); 
                for l = 1:size(csi_phase_matrix,4)
                    csi_phase_matrix2(ii,j,k,l) = corrected_phase(l); 
                end
            end
        end
    end
    
    
    csi_amp_matrix = permute(csi_amp_matrix, [4 3 2 1]);% 1,2,3,4 length*2*3*30
    csi_phase_matrix2 = permute(csi_phase_matrix2, [4 3 2 1]);
    

    for packet_index = 1:length(csi_trace)
        temp = [temp;horzcat(reshape(csi_amp_matrix(:,:,:,packet_index),[1,180]),...
                             reshape(csi_phase_matrix2(:,:,:,packet_index),[1,180]))];
    end
    
    %csvwrite([char(FileName(i)),'.csv'],horzcat(timestamp),temp));
    dlmwrite([char(FileName(i)),'.csv'],horzcat(timestamp,temp),'precision','%.3f');
end
