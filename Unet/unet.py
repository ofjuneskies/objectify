import torch
import torch.nn as nn
import torchvision.models as models



class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()

        # Encoder: Pretrained ResNet-34
        resnet34 = models.resnet34(weights='DEFAULT')
        self.encoder1 = nn.Sequential(*list(resnet34.children())[:3])  # First two layers
        self.encoder2 = nn.Sequential(*list(resnet34.children())[3:5])  # Layer 1
        self.encoder3 = resnet34.layer2  # Layer 2
        self.encoder4 = resnet34.layer3  # Layer 3
        self.encoder5 = resnet34.layer4  # Layer 4

        # Decoder
        self.decoder4 = self._decoder_block(512, 256)
        self.decoder3 = self._decoder_block(256, 128)
        self.decoder2 = self._decoder_block(128, 64)
        self.decoder1 = self._decoder_block(64, 64)

        # Final output layer
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        input_size = x.size()[2:]  # Store original input size (H, W)

        # Encode
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        # Decode
        d4 = self.decoder4(e5) + e4
        d3 = self.decoder3(d4) + e3
        d2 = self.decoder2(d3) + e2
        d1 = self.decoder1(d2) + e1

        # Final output
        out = self.final(d1)

        # Resize output to match input size
        out = nn.functional.interpolate(out, size=input_size, mode='bilinear', align_corners=False)

        return out

num_classes = 15
model = UNet(num_classes=num_classes)

if __name__ == '__main__':
    input_tensor = torch.randn(1, 3, 640, 640)  # Simulated input image
    output = model(input_tensor)
    print("Output shape:", output.shape)  # Expected shape: (batch_size, num_classes, height, width)

    print(model)