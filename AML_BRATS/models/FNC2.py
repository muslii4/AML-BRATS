import torch
import torch.nn as nn
import torch.nn.functional as F

"""
this model is based on The SegNet architecture .
"""
    

class EncoderBlock(nn.Module):
    """Three layered convolutional block with max pooling to decrease the image size
    """
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            return_indices=True
        )

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
        x = self.conv(x)
        size = x.size()
        x, indices = self.pool(x)

        return x, indices, size


class DecoderBlock(nn.Module):
    """Three layered convolutional block with max unpooling to increase the image size
    """
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.unpool = nn.MaxUnpool2d(
            kernel_size=2,
            stride=2
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, indices, output_size) -> torch.Tensor:
        x = self.unpool(x, indices, output_size=output_size)
        x = self.conv(x)

        return x


class SegNet(nn.Module):
    """
    SegNet architecture for semantic segmentation
    using a 5 block encoder-decoder structure with optional Monte Carlo Dropout for uncertainty estimation
    """
    def __init__(self, num_classes=4, dropout_p= 0.2, MC_Dropout=False, num_passes=20) -> None:
        """
        Args: 
            num_classes: Number of output classes for segmentation
            dropout_p: Dropout probability for Monte Carlo Dropout
            MC_Dropout: If True, enables Monte Carlo Dropout for uncertainty estimation
            num_passes: Number of passes for Monte Carlo Dropout

        ATTRIBUTES:
            conv1, conv2, conv3, conv4, conv5: Encoder blocks which uses convolutional layers followed by max pooling
            deconv5, deconv4, deconv3, deconv2, deconv1: Decoder blocks which uses max unpooling (increases the image size) followed by convolutional layers
            final: Final convolutional layer to produce class scores
        """
        super().__init__() 
        self.MC_Dropout = MC_Dropout
        self.dropout_p = dropout_p
        self.num_passes = num_passes
        # Encoder 
        self.conv1 = EncoderBlock(in_channels=4, out_channels=16)
        self.conv2 = EncoderBlock(in_channels=16, out_channels=32)
        self.conv3 = EncoderBlock(in_channels=32, out_channels=64)
        self.conv4 = EncoderBlock(in_channels=64, out_channels=128)
        self.conv5 = EncoderBlock(in_channels=128, out_channels=256)
        # Decoder 
        self.decon5 = DecoderBlock(in_channels=256, out_channels=128)
        self.decon4 = DecoderBlock(in_channels=128, out_channels=64)
        self.decon3 = DecoderBlock(in_channels=64, out_channels=32)
        self.decon2 = DecoderBlock(in_channels=32, out_channels=16)
        self.decon1 = DecoderBlock(in_channels=16, out_channels=16)
    
        self.final = nn.Conv2d(
            in_channels=16,  
            out_channels=num_classes, 
            kernel_size=1
        )

    def _forward(
        self, x: torch.Tensor, force_mc_dropout: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the SegNet

        ARGS:
            x: Input tensor of shape [batch_size, channels, height, width]
        RETURNS:
            Output tensor of shape [batch_size, num_classes, height, width]
        """
        use_mc_dropout = self.MC_Dropout and (self.training or force_mc_dropout)

        # encoder 
        x, idx1, size1 = self.conv1(x)
        if use_mc_dropout:
            x = F.dropout2d(x, p=self.dropout_p, training=True)
        x, idx2, size2 = self.conv2(x)
        if use_mc_dropout:
            x = F.dropout2d(x, p=self.dropout_p, training=True)
        x, idx3, size3 = self.conv3(x)
        if use_mc_dropout:
            x = F.dropout2d(x, p=self.dropout_p, training=True)
        x, idx4, size4 = self.conv4(x)
        if use_mc_dropout:
            x = F.dropout2d(x, p=self.dropout_p, training=True)
        x, idx5, size5 = self.conv5(x)


        # decoder
        if use_mc_dropout:
            x = F.dropout2d(x, p=self.dropout_p, training=True)
        x = self.decon5(x, idx5, size5)
        if use_mc_dropout:
            x = F.dropout2d(x, p=self.dropout_p, training=True)
        x = self.decon4(x, idx4, size4)
        if use_mc_dropout:
            x = F.dropout2d(x, p=self.dropout_p, training=True)
        x = self.decon3(x, idx3, size3)
        if use_mc_dropout:
            x = F.dropout2d(x, p=self.dropout_p, training=True)
        x = self.decon2(x, idx2, size2)
        if use_mc_dropout:
            x = F.dropout2d(x, p=self.dropout_p, training=True)
        x = self.decon1(x, idx1, size1)
    
        x = self.final(x)
        return x
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard call path for training and inference.
        If MC dropout is enabled and the model is in eval mode, this returns
        the mean logits over num_passes stochastic forward passes.
        """
        if self.MC_Dropout and (not self.training) and self.num_passes > 1:
            logits = [
                self._forward(x, force_mc_dropout=True)
                for _ in range(self.num_passes)
            ]
            return torch.stack(logits, dim=0).mean(dim=0)

        return self._forward(x)
        
    def _mc_dropout_forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run multiple forward passes with dropout enabled for uncertainty estimation
        
        ARGS: 
            x: Input tensor
            num_passes: Number of stochastic forward passes to perform
        
        RETURNS:
            mean_pred: Mean prediction across passes
            variance_pred: Variance of predictions across passes
            epistemic_uncertainty: Average variance across classes 
        
        """
        if not self.MC_Dropout:
            raise ValueError("Model must be initialized with MC_Dropout=True")
        
        predictions = []
        self.eval()
        
        with torch.no_grad():
            for _ in range(self.num_passes):
                out = self._forward(x, force_mc_dropout=True)
                predictions.append(torch.softmax(out, dim=1)) 

        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)  
        variance_pred = predictions.var(dim=0)  
        epistemic_uncertainty = variance_pred.mean(dim=1) 
        
        return mean_pred, variance_pred, epistemic_uncertainty

    def predict_with_uncertainty(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convenience wrapper for uncertainty outputs in eval mode."""
        return self._mc_dropout_forward(x)


if __name__ == "__main__":
    model = SegNet(num_classes=4, MC_Dropout=False)
    x = torch.randn(1, 4, 256, 256)
    
    # Regular forward pass 
    out = model(x)
    print(f"Regular output shape: {out.shape}")
    
    # MC Dropout forward pass
    model.MC_Dropout = True
    model.num_passes = 20
    model.eval()
    out_mc = model(x)
    mean, variance, uncertainty = model.predict_with_uncertainty(x)

    print(f"MC-averaged output shape: {out_mc.shape}")
    
    print(f"Uncertainty mean     : {uncertainty.mean().item()}")
    print(f"Uncertainty max      : {uncertainty.max().item()}")
    print(f"Uncertainty min      : {uncertainty.min().item()}")
    print(f"Uncertainty variance : {uncertainty.var().item()}")



#----------------------------------------------------------
#METRICS

def dice_score(pred, target, smooth=1e-6):
    """
    For overlap between prediction and true tumour
    """
    pred = torch.argmax(pred, dim=1)
    pred = (pred > 0).float()
    target = (target > 0).float()
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    
    dice = ((2.0 * intersection + smooth)/ (pred.sum() + target.sum() + smooth) )
    return dice.item()
    
def iou_score(pred, target, smooth=1e-6):
    """
    How much predicted region matches real region
    """
    pred = torch.argmax(pred, dim=1)
    pred = (pred > 0).float()
    target = (target > 0).float()
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()



