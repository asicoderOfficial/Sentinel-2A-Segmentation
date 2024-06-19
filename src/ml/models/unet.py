import torch
import torch.nn as nn



class UNet(nn.Module):

    def __init__(self, in_channels:int, out_channels: int, n_classes:int=2) -> None:
        """ Implements the basic U-Net model from the paper 'U-Net: Convolutional Networks for Biomedical Image Segmentation'.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            n_classes (int): The number of classes to segment. Defaults to 2 (binary segmentation).
        
        Returns:
            None
        
        Notes:
            - The decoder features include more semantic features, such as 'in this area there is a building'.
            - The encoder features contain more specific features, such as 'these exactly are the pixels where the building is at'.
            - When combining them, the pixel-perfect segmentation happens.
            - Same convolution: output feature map has the same spatial dimensions as the input.
            - Upsample blocks have 2x the previous conv_layer out_channels, because of the skip connections, that add the encoder + decoder feature maps together (concatenation).
        """        
        super(UNet, self).__init__()

        self.n_classes = n_classes

        # Encoder: downsampling
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1), # Same convolution
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # Same convolution
            nn.ReLU()
        )
        self.conv_layer_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), # This downsampling operation, doubles the channels
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv_layer_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), # This downsampling operation, doubles the channels
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv_layer_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), # This downsampling operation, doubles the channels
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # Bottleneck: from encoder to decoder
        self.conv_layer_5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), # This downsampling operation, doubles the channels
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # Decoder: upsample.
        self.upsample_4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.conv_layer_6 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.upsample_3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.conv_layer_7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.upsample_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv_layer_8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.upsample_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv_layer_9 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # Output layer
        self.output_layer = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x) -> torch.Tensor:
        # Encoder
        enc1 = self.conv_layer_1(x)
        enc2 = self.conv_layer_2(enc1)
        enc3 = self.conv_layer_3(enc2)
        enc4 = self.conv_layer_4(enc3)
        # Bottleneck
        bottleneck = self.conv_layer_5(enc4)
        # Decoder
        dec4 = self.upsample_4(bottleneck)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.conv_layer_6(dec4)
        dec3 = self.upsample_3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.conv_layer_7(dec3)
        dec2 = self.upsample_2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.conv_layer_8(dec2)
        dec1 = self.upsample_1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.conv_layer_9(dec1)
        # Output layer
        output = self.output_layer(dec1)

        return output
        