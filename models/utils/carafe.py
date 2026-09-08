import torch
import torch.nn as nn
import torch.nn.functional as F

def carafe_forward(
    features: torch.Tensor,
    masks: torch.Tensor,
    kernel_size: int,
    group_size: int,
    scale_factor: int
) -> torch.Tensor:
    """
    Pure-PyTorch implementation of the CARAFE upsampling operator.
    
    Args:
        features (Tensor): Input feature map of shape (N, C, H, W).
        masks (Tensor): Reassembly kernel weights of shape 
            (N, kernel_size*kernel_size*group_size, H_out, W_out),
            where H_out = H*scale_factor and W_out = W*scale_factor.
        kernel_size (int): The spatial size of the reassembly kernel.
        group_size (int): The group size to divide channels. Must divide C.
        scale_factor (int): The upsampling factor.
    
    Returns:
        Tensor: Upsampled feature map of shape (N, C, H*scale_factor, W*scale_factor).
    """
    N, C, H, W = features.size()
    out_H, out_W = H * scale_factor, W * scale_factor
    num_channels = C // group_size  # channels per group

    # Reshape features to (N, group_size, num_channels, H, W)
    features = features.view(N, group_size, num_channels, H, W)
    # Merge batch and group dims for unfolding
    features_reshaped = features.view(N * group_size, num_channels, H, W)
    # Extract local patches; use padding so that output spatial dims match input
    patches = F.unfold(features_reshaped, kernel_size=kernel_size, 
                       padding=(kernel_size - 1) // 2)
    # patches shape: (N*group_size, num_channels*kernel_size*kernel_size, H*W)
    # Reshape to (N, group_size, num_channels, kernel_size*kernel_size, H, W)
    patches = patches.view(N, group_size, num_channels, kernel_size * kernel_size, H, W)
    # Flatten spatial dimensions: now (N, group_size, num_channels, kernel_size*kernel_size, H*W)
    patches = patches.view(N, group_size, num_channels, kernel_size * kernel_size, H * W)

    # For each output pixel location, determine the corresponding base input index.
    # For an output coordinate (oh, ow), the corresponding input index is:
    #   h = oh // scale_factor, w = ow // scale_factor, linear index = h * W + w.
    device = features.device
    # Create coordinate indices for output
    h_idx = torch.div(torch.arange(out_H, device=device), scale_factor, rounding_mode='floor')  # (out_H,)
    w_idx = torch.div(torch.arange(out_W, device=device), scale_factor, rounding_mode='floor')  # (out_W,)
    # Form a 2D grid of base indices (shape: out_H x out_W)
    h_idx = h_idx.unsqueeze(1).expand(out_H, out_W)  # (out_H, out_W)
    w_idx = w_idx.unsqueeze(0).expand(out_H, out_W)  # (out_H, out_W)
    base_idx = (h_idx * W + w_idx).view(-1)  # (out_H*out_W,)

    # Expand base_idx so that it can index the last dimension of patches:
    # Desired shape for gathering: (N, group_size, num_channels, kernel_size*kernel_size, out_H*out_W)
    base_idx = base_idx.view(1, 1, 1, 1, -1).expand(N, group_size, num_channels, kernel_size * kernel_size, -1)
    # Gather patches corresponding to each output location
    gathered_patches = torch.gather(patches, -1, base_idx)
    # Reshape gathered patches to (N, group_size, num_channels, kernel_size*kernel_size, out_H, out_W)
    gathered_patches = gathered_patches.view(N, group_size, num_channels, kernel_size * kernel_size, out_H, out_W)

    # Reshape masks to separate groups.
    # Expected mask shape: (N, kernel_size*kernel_size*group_size, out_H, out_W)
    # Reshape to: (N, group_size, kernel_size*kernel_size, out_H, out_W)
    masks = masks.view(N, group_size, kernel_size * kernel_size, out_H, out_W)
    # For multiplication, add a channel dimension so that masks shape becomes
    # (N, group_size, 1, kernel_size*kernel_size, out_H, out_W)
    masks = masks.unsqueeze(2)
    # Expand masks to match gathered_patches: (N, group_size, num_channels, kernel_size*kernel_size, out_H, out_W)
    masks = masks.expand(-1, -1, num_channels, -1, -1, -1)

    # Multiply patches with masks and sum over the kernel dimension.
    # This yields the reassembled features for each output location.
    out = (gathered_patches * masks).sum(dim=3)  # shape: (N, group_size, num_channels, out_H, out_W)
    # Reshape back to (N, C, out_H, out_W)
    out = out.view(N, C, out_H, out_W)
    return out


class CARAFE(nn.Module):
    """
    CARAFE: Content-Aware ReAssembly of Features

    This PyTorch module implements the CARAFE upsampling operator in pure Python.
    Given an input feature map and its corresponding reassembly masks, the module
    reassembles features from local patches to produce a higher-resolution output.

    Args:
        kernel_size (int): Reassembly kernel size.
        group_size (int): Group size for channel grouping (must divide number of channels).
        scale_factor (int): Upsample ratio.
    """
    def __init__(self, kernel_size: int, group_size: int, scale_factor: int):
        super(CARAFE, self).__init__()
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.scale_factor = scale_factor

    def forward(self, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return carafe_forward(features, masks, self.kernel_size, self.group_size, self.scale_factor)


class CARAFEPack(nn.Module):
    """
    A unified package of the CARAFE upsampler that contains:
    1) A channel compressor.
    2) A content encoder that predicts reassembly masks.
    3) The CARAFE operator.

    This is modeled after the official CARAFE package.

    Args:
        channels (int): Number of input feature channels.
        scale_factor (int): Upsample ratio.
        up_kernel (int): Kernel size for the CARAFE operator.
        up_group (int): Group size for the CARAFE operator.
        encoder_kernel (int): Kernel size of the content encoder.
        encoder_dilation (int): Dilation rate for the content encoder.
        compressed_channels (int): Output channels for the channel compressor.
    """
    def __init__(
        self,
        channels: int,
        scale_factor: int,
        up_kernel: int = 5,
        up_group: int = 1,
        encoder_kernel: int = 3,
        encoder_dilation: int = 1,
        compressed_channels: int = 64
    ):
        super(CARAFEPack, self).__init__()
        self.channels = channels
        self.scale_factor = scale_factor
        self.up_kernel = up_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels

        # Compress input channels.
        self.channel_compressor = nn.Conv2d(channels, compressed_channels, kernel_size=1)
        # Predict reassembly masks.
        self.content_encoder = nn.Conv2d(
            compressed_channels,
            up_kernel * up_kernel * up_group * scale_factor * scale_factor,
            kernel_size=encoder_kernel,
            padding=int((encoder_kernel - 1) * encoder_dilation / 2),
            dilation=encoder_dilation
        )
        # Initialize weights (using Xavier for conv layers).
        nn.init.xavier_uniform_(self.channel_compressor.weight)
        nn.init.xavier_uniform_(self.content_encoder.weight)
        if self.channel_compressor.bias is not None:
            nn.init.constant_(self.channel_compressor.bias, 0)
        if self.content_encoder.bias is not None:
            nn.init.constant_(self.content_encoder.bias, 0)

    def kernel_normalizer(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Normalize and reshape the mask.
        Applies pixel shuffle to upsample the predicted kernel weights and then
        applies softmax normalization across the kernel dimension.
        
        Args:
            mask (Tensor): Predicted mask of shape (N, out_channels, H, W).
            
        Returns:
            Tensor: Normalized mask of shape (N, up_group * up_kernel^2, H*scale, W*scale).
        """
        # Pixel shuffle to rearrange and upsample the mask.
        mask = F.pixel_shuffle(mask, self.scale_factor)
        N, mask_c, H, W = mask.size()
        # Determine the number of channels per kernel
        mask_channel = mask_c // (self.up_kernel ** 2)
        mask = mask.view(N, mask_channel, self.up_kernel ** 2, H, W)
        mask = F.softmax(mask, dim=2)
        mask = mask.view(N, mask_channel * self.up_kernel ** 2, H, W).contiguous()
        return mask

    def feature_reassemble(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return carafe_forward(x, mask, self.up_kernel, self.up_group, self.scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compressed_x = self.channel_compressor(x)
        mask = self.content_encoder(compressed_x)
        mask = self.kernel_normalizer(mask)
        out = self.feature_reassemble(x, mask)
        return out


# === Example Usage ===
if __name__ == '__main__':
    # Create dummy input: batch size 2, 64 channels, 32x32 spatial resolution.
    x = torch.randn(2, 64, 32, 32).cuda()  # assuming GPU available
    # Define CARAFEPack with upsample ratio 2.
    # For example, use kernel size 5, group size 1.
    upsampler = CARAFEPack(channels=64, scale_factor=2, up_kernel=5, up_group=1).cuda()
    # Get upsampled feature map.
    out = upsampler(x)
    print("Input shape: ", x.shape)
    print("Output shape:", out.shape)  # Expected shape: (2, 64, 64, 64)
