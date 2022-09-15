#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

#%%
class CentroidPredictionLSTM(nn.Module):
    """Long-short-term-memory network for supervised prediction of centroid positions. 

    Args:
        input_features (int): number of input features at each time step
        hidden_features (int): number of features of LSTM hidden state
        output_features (int): number of output features at each time step
        num_layers (int): number of LSTM hidden layers
        batch_size (int): number of data patterns to be fed to network simultaneously
        seq_len_in (int): length of input window
        seq_len_out (int): length of predicted window
        device (torch.device): torch cuda device
        dropout (float, optional): probability of dropout [0,1] in dropout layer  
        bi (bool, optional): if True, becomes a bidirectional LSTM
    """
    def __init__(self, input_features, hidden_features, output_features,
                 num_layers, seq_len_in, seq_len_out, 
                 device, dropout=0, bi=False):
        super(CentroidPredictionLSTM, self).__init__()
        
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        self.output_features = output_features
        self.bi = bi
        self.device = device
        
        # construct lstm 
        self.lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_features,
                            num_layers=num_layers, dropout=dropout, 
                            bidirectional=self.bi, batch_first=True)
        
        # construct fully-connected layer
        if self.bi is False:
            self.fc = nn.Linear(in_features=hidden_features, 
                                out_features=seq_len_out * output_features)
        if self.bi:
            self.fc = nn.Linear(in_features=hidden_features * 2, 
                                out_features=seq_len_out * output_features)        
        
    def reset_h_c_states(self, batch_size=1):
        "Reset the hidden state and the cell state of the LSTM."
        
        # tensors containing the initial hidden state and initial cell state
        # with shape (num_layers * num_directions, batch_size, hidden_size)
        if self.bi is False:
            self.h_c = (torch.zeros(self.num_layers, batch_size, self.hidden_features),
                        torch.zeros(self.num_layers, batch_size, self.hidden_features))
        if self.bi:
            self.h_c = (torch.zeros(self.num_layers * 2, batch_size, self.hidden_features),
                        torch.zeros(self.num_layers * 2, batch_size, self.hidden_features))           

        # move states to cuda device
        self.h_c = (self.h_c[0].to(self.device), self.h_c[1].to(self.device))
            

    def forward(self, input_batch):
        """Compute forward pass through network.

        Args:
            input_batch (array): input array with shape (batch_size, seq_len_in, input_features)   

        Returns:
            predictions (array): output array with shape (batch_size, seq_len_out, output_features) 
                                    containing predicted time sequence
        """
        # get batch size from current input batch
        batch_size = input_batch.shape[0]
        
        # reset hidden state and cell state for current batch of data
        self.reset_h_c_states(batch_size=batch_size)
        # print(f'Shape of input_batch: {input_batch.shape} ')  # (batch_size, seq_len_in, input_features) 
               
        # propagate input of shape=(batch, seq_len, input_size) through LSTM
        lstm_out, self.h_c = self.lstm(input_batch, self.h_c)
        # print(f'Shape of lstm_out: {lstm_out.shape} ')  # (batch_size, seq_len_in, hidden_features)

        # only take the output of the last LSTM module
        # (can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction)
        predictions = self.fc(lstm_out[:, -1, :])
        
        # reshape to explicitly get output_features dimension back
        predictions = predictions.reshape(batch_size, self.seq_len_out, self.output_features)
        # print(f'Shape of predictions: {predictions.shape} ')   # (batch_size, seq_len_out, output_features)
        
        # return output windows of batch
        return predictions
 
 
class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell proposed by Shi et al. (2015). 
    Code inspired by: https://github.com/jhhuang96/ConvLSTM-PyTorch
    
    Args:
        channels_in (int): number of input channels at each time step
        channels_out (int): number of output channels at each time step
        seq_len (int): number of ConvLSTM cells, correlated to input/ouput sequence length
        filter_size (int): size of convolutional kernel
        device (torch.device ): torch cuda device
        sampling (str, optional): whether to use zeros or previous output as input to the cells
        group_norm (bool, optional): whether to use group norm or not
    """
    def __init__(self, channels_in, channels_out, seq_len,
                 filter_size, device, sampling=None, group_norm=False):
        super(ConvLSTMCell, self).__init__()

        self.sampling = sampling
        self.device = device
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.seq_len = seq_len
        self.filter_size = filter_size
        # to ensure that the output has the same size
        self.padding = (filter_size - 1) // 2
        if group_norm is False:
            self.conv = nn.Conv2d(self.channels_in + self.channels_out,
                        4 * self.channels_out, self.filter_size, 1,
                        self.padding)
        else:
            self.conv = nn.Sequential(
                                nn.Conv2d(self.channels_in + self.channels_out,
                                4 * self.channels_out, self.filter_size, 1,
                                self.padding),
                                nn.GroupNorm(4 * self.channels_out // 32, 4 * self.channels_out))

    def forward(self, inputs, hidden_state=None):
        """Compute forward pass through network.

        Args:
            inputs (tensor): input tensor with shape (seq_len_in, batch_size, channels_in, height, width).
            hidden_state (tensor, optional): input hidden state with shape (batch_size, channels_out, height, width)

        Returns:
            tensor: ouput tensor with shape (seq_len_out, batch_size, channels_out, height, width)
        """
        if hidden_state is None:
            # reset hidden and cell states
            hx = torch.zeros(inputs.shape[1], self.channels_out,
                             inputs.shape[-2], inputs.shape[-1]).to(self.device)
            cx = torch.zeros(inputs.shape[1], self.channels_out,
                             inputs.shape[-2], inputs.shape[-1]).to(self.device)
        else:
            hx, cx = hidden_state
            if hx.shape[1] != self.channels_out:
                raise ValueError("Channel dimension for hidden_state must match channels_out!")            
            
        output_inner = []
        for index in range(self.seq_len):
            if self.sampling is None:
                # be aware that sequence len of input data must match self.seq_len
                x = inputs[index, ...]  
            elif self.sampling == 'always_sampling':
                if index == 0:
                    # first input to ConvLSTM decoder shall be last spatial decoder output
                    x = inputs[-1, ...] 
                else:
                    # successive input to convLSTM decoder shall be output of previous cells
                    x = output_inner[-1]
            else:
                raise Exception('Unknown sampling specified!')
            
            # print(f'Shape x: {x.shape}')  # e.g. torch.Size([32, 64, 55, 55])
            # print(f'Shape hx: {hx.shape}')  # e.g. torch.Size([32, 64, 55, 55])
            combined = torch.cat((x, hx), 1)
            # print(f'Shape of combined: {combined.shape}')  # e.g. torch.Size([32, 128, 55, 55])
            gates = self.conv(combined)  
            # print(f'Shape of gates: {gates.shape}') # batch_size, channels_out*4, h, w
            # it should return 4 tensors: i,f,g,o
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.channels_out, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)   #  seq_len_in, batch_size, channels_out, height, width
            hx = hy    #  batch_size, channels_out, height, width
            cx = cy
        return torch.stack(output_inner), (hy, cy)
    

class SegmentationPredictionConvLSTM(nn.Module):
    """Convolutional LSTM inspired by Sautermeister (2016). Overall architecture:
        - Econding branch via convolutions outputs feature maps encoding the input data
        - Convolutional LSTM cells at the bottom take into accout for temporal nature of data
        - Decoding branch upsamples to predicted segmentations

    Args:
        input_shape (array): np.shape of input tensor
        seq_len_out (int): length of predicted window
        device (torch.device): torch cuda device
        dec_input_init_zeros (bool, optional): whether to initialize input of decoder with tensor of zeros 
    """
    def __init__(self, input_shape, seq_len_out, device, dec_input_init_zeros=False):
        super(SegmentationPredictionConvLSTM, self).__init__()
        
        # get input image height and width etc
        _, self.seq_len_in, self.channels_in, self.height, self.width = input_shape
        self.seq_len_out = seq_len_out
        self.device = device
        self.dec_input_init_zeros = dec_input_init_zeros
        
        # encoder
        self.conv_kernel1 = 5
        self.conv_kernel2 = 3
        self.conv_enc1 = nn.Sequential(nn.Conv2d(in_channels=self.channels_in, out_channels=32, kernel_size=self.conv_kernel1, stride=2, padding=(self.conv_kernel1-1)//2), 
                                    nn.ReLU(), 
                                    nn.BatchNorm2d(32))
        self.conv_enc2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.conv_kernel2, stride=1, padding=(self.conv_kernel2-1)//2), 
                                    nn.ReLU(),
                                    nn.BatchNorm2d(64))
        self.conv_enc3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.conv_kernel2, stride=2, padding=(self.conv_kernel2-1)//2), 
                                    nn.ReLU(),
                                    nn.BatchNorm2d(64))
        self.convlstm_enc = ConvLSTMCell(channels_in=64, channels_out=64, seq_len=self.seq_len_in,
                                          filter_size=5, device=self.device, sampling=None)
        
        # decoder
        self.convlstm_dec = ConvLSTMCell(channels_in=64, channels_out=64, seq_len=self.seq_len_out,
                                          filter_size=5, device=self.device, sampling='always_sampling')
        self.upconv_dec1 = nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=self.conv_kernel2, stride=2, padding=(self.conv_kernel2-1)//2),
                                        nn.ReLU(), 
                                        nn.BatchNorm2d(64))
        self.upconv_dec2 = nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.conv_kernel2, stride=1, padding=(self.conv_kernel2-1)//2),
                                        nn.ReLU(), 
                                        nn.BatchNorm2d(32))
        self.upconv_dec3 = nn.Sequential(nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=self.conv_kernel1, stride=2, 
                                                            padding=(self.conv_kernel1-1)//2))
        
    
    def forward(self, x):
        # swap batch and sequence dimension for forward pass
        x_swap = x.transpose(1,0) # s, b, c, h, w
        # merge time dimension into batch dimension for encoder conv layers
        x_enc = x_swap.reshape(-1, self.channels_in, self.height, self.width)  # s*b, c, h, w
        
        # encoder
        x_enc1 = self.conv_enc1(x_enc)
        x_enc2 = self.conv_enc2(x_enc1)
        x_enc3 = self.conv_enc3(x_enc2)
        
        # reshape inputs to explicitly include time dimension
        x_enc3_reshaped = x_enc3.reshape(self.seq_len_in, x.shape[0], x_enc3.shape[-3],
                                        x_enc3.shape[-2], x_enc3.shape[-1])
        # print(f'Shape of x_enc3 prior rnn: {x_enc3_reshaped.shape}') #  (s, b, c, ...)
        _, states_enc = self.convlstm_enc(inputs=x_enc3_reshaped, hidden_state=None)

        if self.dec_input_init_zeros:
            # intiliaze time sequence with zeros for decoder
            x_dec_init = torch.zeros(self.seq_len_out, x.shape[0], x_enc3.shape[-3],
                                    x_enc3.shape[-2], x_enc3.shape[-1]).to(self.device)   
        else:
            x_dec_init = x_enc3_reshaped
        
        # decoder taking over states from encoder
        outputs_dec, _  = self.convlstm_dec(inputs=x_dec_init, hidden_state=states_enc)
        # print(f'Shape of outputs_dec: {outputs_dec.shape}') 

        # merge time dimension into batch dimension for conv layers
        outputs_dec = outputs_dec.reshape(-1, outputs_dec.shape[-3], outputs_dec.shape[-2], outputs_dec.shape[-1]) # s*b, c, h, w
             
        x_dec1 = self.upconv_dec1(outputs_dec)
        x_dec1 = F.pad(input=x_dec1, pad=(0, 1, 0, 1), mode='constant', value=0.0)
        
        x_dec2 = self.upconv_dec2(x_dec1)

        x_dec3 = self.upconv_dec3(x_dec2)
        x_dec3 = F.pad(input=x_dec3, pad=(0, 1, 0, 1), mode='constant', value=0.0)
        
        # reshape inputs to explicitly include time dimension --> b, s, c, h, w
        x_dec3 = x_dec3.reshape(self.seq_len_out, x.shape[0], x_dec3.shape[-3],
                                        x_dec3.shape[-2], x_dec3.shape[-1])
        
        # swap batch and sequence dim back
        x_dec3 =  x_dec3.transpose(1, 0) # b, s, c, h, w
        
        # apply sigmoid to get logits
        predictions = torch.sigmoid(x_dec3)
        
        return predictions


class SegmentationPredictionConvLSTMSTL(nn.Module):
    """Convolutional LSTM + Spatial Transforer Layer inspired by Romaguera et al. (2019). Overall architecture:
        - Econding branch via convolutions outputs feature maps encoding the input data
        - Convolutional LSTM cells at the bottom take into accout for temporal nature of data
        - Decoding branch upsamples the predicted deformation vector fields
        - STL warps input frame with predicted deformation to build predicted frames 

    Args:
        input_shape (array): np.shape of input tensor
        seq_len_out (int): length of predicted window
        device (torch.device): torch cuda device
        max_displacement (float, optional): cut-off value in terms of relative image width/height for displacements
        dec_input_init_zeros (bool, optional): whether to initialize input of decoder with tensor of zeros 
        input_data (str, optional): input data for network, eg 'segmentations' or 'segmentations_and_frames'.
    """
    def __init__(self, input_shape, seq_len_out, device, max_displacement=0.2, 
                 dec_input_init_zeros=False, input_data='segmentations'):
        super(SegmentationPredictionConvLSTMSTL, self).__init__()
        
        # get input image height and width etc
        _, self.seq_len_in, self.channels_in, self.height, self.width = input_shape
        self.seq_len_out = seq_len_out
        self.device = device
        self.dec_input_init_zeros = dec_input_init_zeros
        self.input_data = input_data
        
        # encoder 
        self.conv_kernel = 5
        self.maxpool_kernel = 2
        self.conv_enc1 = nn.Sequential(nn.Conv2d(in_channels=self.channels_in, out_channels=16, kernel_size=self.conv_kernel, stride=1), 
                                    nn.BatchNorm2d(16), 
                                    nn.ReLU(), 
                                    nn.MaxPool2d(kernel_size=self.maxpool_kernel, stride=1))
        self.conv_enc2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.conv_kernel, stride=1), 
                                    nn.BatchNorm2d(32), 
                                    nn.ReLU(), 
                                    nn.MaxPool2d(kernel_size=self.maxpool_kernel, stride=1))
        self.conv_enc3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.conv_kernel, stride=1), 
                                    nn.BatchNorm2d(64), 
                                    nn.ReLU(), 
                                    nn.MaxPool2d(kernel_size=self.maxpool_kernel, stride=1))
        self.convlstm_enc = ConvLSTMCell(channels_in=64, channels_out=64, seq_len=self.seq_len_in,
                                          filter_size=5, device=self.device, sampling=None)
        
        # decoder
        self.convlstm_dec = ConvLSTMCell(channels_in=64, channels_out=64, seq_len=self.seq_len_out,
                                          filter_size=5, device=self.device, sampling='always_sampling')
        self.conv_dec1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.conv_kernel, stride=1, padding=(self.conv_kernel-1)//2), 
                                    nn.BatchNorm2d(64))
        self.upconv_dec1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=self.conv_kernel, stride=1, padding=0)      
        self.conv_dec2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=self.conv_kernel, stride=1, padding=(self.conv_kernel-1)//2), 
                                    nn.BatchNorm2d(32))
        self.upconv_dec2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=self.conv_kernel, stride=1, padding=0) 
        self.conv_dec3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=16, kernel_size=self.conv_kernel, stride=1, padding=(self.conv_kernel-1)//2),
                                    nn.BatchNorm2d(16))
        self.upconv_dec3 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=self.conv_kernel, stride=1, padding=0)        
        self.conv_dec4 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=self.conv_kernel, stride=1, padding=(self.conv_kernel-1)//2)
        self.conv_dec5 = nn.Conv2d(in_channels=8, out_channels=2, kernel_size=self.conv_kernel, stride=1, padding=(self.conv_kernel-1)//2)
        
        # max displacement of DVF
        self.max_displacement = max_displacement
        
        # generate identity displacement for combination with computed displacement 
        h_s = torch.linspace(-1, 1, self.height)  # needs to be normalized betwwen -1 and +1 for grid_sample function
        w_s = torch.linspace(-1, 1, self.width)
        h_s, w_s = torch.meshgrid([h_s, w_s])
        mesh_grid = torch.stack([w_s, h_s])
        self.mesh_grid = mesh_grid.permute(1, 2, 0).to(device)  # h x w x 2
            
    def warp_image(self, img, grid):
        # use grid_sample Pytorch function to warp image
        wrp = F.grid_sample(img, grid, mode='bilinear')
        return wrp
    
    def forward(self, x):
        # swap batch and sequence dimension for forward pass
        x_swap = x.transpose(1,0) # s, b, c, h, w
        
        # merge time dimension into batch dimension for encoder conv layers
        x_enc = x_swap.reshape(-1, self.channels_in, self.height, self.width)  # s*b, c, h, w
        
        # encoder
        x_enc1 = self.conv_enc1(x_enc)
        x_enc2 = self.conv_enc2(x_enc1)
        x_enc3 = self.conv_enc3(x_enc2)
        
        # reshape inputs to explicitly include time dimension
        x_enc3_reshaped = x_enc3.reshape(self.seq_len_in, x.shape[0], x_enc3.shape[-3],
                                        x_enc3.shape[-2], x_enc3.shape[-1])
        # print(f'Shape of x_enc3 prior rnn: {x_enc3_reshaped.shape}') #  torch.Size([s, b, c, ...])
        _, states_enc = self.convlstm_enc(inputs=x_enc3_reshaped, hidden_state=None)

        if self.dec_input_init_zeros:
            # intiliaze input of decoder with zeros
            x_dec_init = torch.zeros(self.seq_len_out, x.shape[0], x_enc3.shape[-3],
                                    x_enc3.shape[-2], x_enc3.shape[-1]).to(self.device) 
        else:
            # take the last element of the encoder as first input and 
            # sample the output of previous cells for the subsequent cells
            x_dec_init = x_enc3_reshaped
                    
        # decoder taking over states from encoder
        outputs_dec, _  = self.convlstm_dec(inputs=x_dec_init, hidden_state=states_enc)
        # print(f'Shape of outputs_dec: {outputs_dec.shape}') # e.g. torch.Size([10, 32, 64, 49, 49])

        # merge time dimension into batch dimension for conv layers
        outputs_dec = outputs_dec.reshape(-1, outputs_dec.shape[-3], outputs_dec.shape[-2], outputs_dec.shape[-1])  # s*b, c, h, w

        x_dec1 = self.upconv_dec1(self.conv_dec1(outputs_dec))
        x_dec1 = F.pad(input=x_dec1, pad=(0, 1, 0, 1), mode='constant', value=0.0)
        
        x_dec2 = self.upconv_dec2(self.conv_dec2(x_dec1))
        x_dec2 = F.pad(input=x_dec2, pad=(0, 1, 0, 1), mode='constant', value=0.0)

        x_dec3 = self.upconv_dec3(self.conv_dec3(x_dec2))
        x_dec3 = F.pad(input=x_dec3, pad=(0, 1, 0, 1), mode='constant', value=0.0)

        x_dec4 = self.conv_dec4(x_dec3)
        
        x_dec5 = self.conv_dec5(x_dec4)
        # print(f'Shape of x_dec5: {x_dec5.shape}')  # e.g. torch.Size([320, 2, 64, 64])
        
        # reshape inputs to explicitly include time dimension
        x_dec5 = x_dec5.reshape(self.seq_len_out, x.shape[0], x_dec5.shape[-3],
                                        x_dec5.shape[-2], x_dec5.shape[-1])
        
        # move channels to last dim
        displacements = torch.moveaxis(x_dec5, 2, -1)  
        # print(f'Shape of displacements: {displacements.shape}') # (s, b, h, w, 2)
        
        for output_frame_nr in range(self.seq_len_out):
            displacement = displacements[output_frame_nr, ...]
            # print(f'Shape of displacement: {displacement.shape}') # (b, h, w, 2)
        
            # squash displacement into -1 and +1 for compatibility with F.grid_sample
            computed_displacement = torch.tanh(displacement) * self.max_displacement
            # print(f'max computed displacement: {torch.max(computed_displacement)}')
            # print(f'min computed displacement: {torch.min(computed_displacement)}')
            # add normalized computed displacement to identity meshgrid
            displacement_grid = computed_displacement + self.mesh_grid.unsqueeze(0)

            if output_frame_nr == 0:
                # warp last input digit using displacement grid
                if self.input_data == 'segmentations_and_frames':
                    warped_seg = self.warp_image(x[:, -1, 0, None, ...], displacement_grid)
                else:
                    warped_seg = self.warp_image(x[:, -1, ...], displacement_grid)
                # print(f'Shape of warped_seg: {np.shape(warped_seg)}')  # (b, c, h, w)
                
                # define predictions etc tensor and add sequence dimension
                predictions = warped_seg[:, None, ...] 
                computed_displacements = computed_displacement[:, None, ...]
                displacement_grids = displacement_grid[:, None, ...]
                
            else:
                # warp last predicted digit using displacement grid
                warped_seg = self.warp_image(predictions[:, -1, ...], displacement_grid)
                # print(f'Shape of warped_seg: {np.shape(warped_seg)}')  # (b, c, h, w)
                
                # concatenate the first predicted frame with the successive ones
                predictions = torch.cat((predictions, warped_seg[:, None, ...]), dim=1)      
                # print(f'Shape of predictions: {predictions.shape}')  # e.g. torch.Size([32, 1 to 10, 1, 64, 64])    
                computed_displacements = torch.cat((computed_displacements, computed_displacement[:, None, ...]), dim=1)      
                displacement_grids = torch.cat((displacement_grids, displacement_grid[:, None, ...]), dim=1)      
        
        return predictions, displacement_grids, computed_displacements
   
    

