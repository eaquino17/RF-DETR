import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import MultiheadAttention
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class ReceptiveFieldEnhancement(nn.Module):
    """Receptive Field Enhancement Module for RF-DETR"""
    def __init__(self, d_model, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.d_model = d_model
        
        # Multi-scale convolutions
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, dilation=1),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=4, dilation=4)
        ])
        
        # Feature fusion
        self.fusion_conv = nn.Conv2d(d_model * num_scales, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x shape: [batch_size, d_model, H, W]
        batch_size, d_model, H, W = x.shape
        
        # Apply multi-scale convolutions
        multi_scale_features = []
        for conv in self.conv_layers:
            feature = F.relu(conv(x))
            multi_scale_features.append(feature)
        
        # Concatenate and fuse features
        fused = torch.cat(multi_scale_features, dim=1)
        enhanced = self.fusion_conv(fused)
        
        # Add residual connection
        enhanced = enhanced + x
        
        # Reshape for layer norm: [batch_size, H*W, d_model]
        enhanced = enhanced.flatten(2).transpose(1, 2)
        enhanced = self.norm(enhanced)
        
        return enhanced

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, src, src_key_padding_mask=None):
        return self.encoder(src, src_key_padding_mask=src_key_padding_mask)

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        return self.decoder(tgt, memory, tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=memory_key_padding_mask)

class RFDETR(nn.Module):
    def __init__(self, num_classes, num_queries=100, d_model=256, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model
        
        # CNN Backbone (ResNet-50)
        backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Reduce channel dimension
        self.input_proj = nn.Conv2d(2048, d_model, kernel_size=1)
        
        # Receptive Field Enhancement Module
        self.rf_enhancement = ReceptiveFieldEnhancement(d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer
        self.encoder = TransformerEncoder(d_model, nhead, num_encoder_layers)
        self.decoder = TransformerDecoder(d_model, nhead, num_decoder_layers)
        
        # Object queries
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # Prediction heads
        self.class_head = nn.Linear(d_model, num_classes + 1)  # +1 for background
        self.bbox_head = nn.Linear(d_model, 4)
        
    def forward(self, images):
        batch_size = images.shape[0]
        
        # Extract features using backbone
        features = self.backbone(images)  # [batch_size, 2048, H, W]
        features = self.input_proj(features)  # [batch_size, d_model, H, W]
        
        # Apply Receptive Field Enhancement
        enhanced_features = self.rf_enhancement(features)  # [batch_size, H*W, d_model]
        
        # Add positional encoding
        enhanced_features = self.pos_encoding(enhanced_features.transpose(0, 1)).transpose(0, 1)
        
        # Transformer encoder
        memory = self.encoder(enhanced_features)  # [batch_size, H*W, d_model]
        
        # Object queries
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Transformer decoder
        decoder_output = self.decoder(query_embed, memory)  # [batch_size, num_queries, d_model]
        
        # Predictions
        class_logits = self.class_head(decoder_output)  # [batch_size, num_queries, num_classes+1]
        bbox_coords = self.bbox_head(decoder_output).sigmoid()  # [batch_size, num_queries, 4]
        
        return {
            'pred_logits': class_logits,
            'pred_boxes': bbox_coords
        }

def build_rf_detr(num_classes, num_queries=100):
    """Build RF-DETR model"""
    model = RFDETR(
        num_classes=num_classes,
        num_queries=num_queries,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6
    )
    return model

if __name__ == "__main__":
    # Test model
    model = build_rf_detr(num_classes=1)  # Car detection (1 class)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 512, 512)
    output = model(dummy_input)
    
    print(f"Class logits shape: {output['pred_logits'].shape}")
    print(f"Bbox coords shape: {output['pred_boxes'].shape}")
