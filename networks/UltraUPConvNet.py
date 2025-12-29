from matplotlib.sankey import UP
import torch
import torch.nn as nn
import torch.nn.functional as F
from .convnext import convnext_tiny
#from .decoder_UPerhead import UPerHead

class UltraUPConvNet(nn.Module):
    def __init__(self, args, prompt=False):
        super().__init__()
        self.prompt = prompt
        # ConvNeXt backbone
        if args.use_pretrained_model:
            self.backbone = convnext_tiny(pretrained=False, num_classes=1000)
            checkpoint = torch.load(args.pretrained_path, map_location='cpu')
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
            self.backbone.load_state_dict(state_dict, strict=False)
        else:
            self.backbone = convnext_tiny(pretrained=False, num_classes=1000)
            print("No pretrained weights provided for ConvNeXt.")
        
        embed_dim = 768
        if prompt:
            self.dec_prompt_mlp = nn.Linear(15, embed_dim)
            self.prompt_proj_layers = nn.ModuleList([
                nn.Linear(15, 96),
                nn.Linear(15, 192),
                nn.Linear(15, 384),
                nn.Linear(15, 768),
            ])
            # 为每个金字塔层单独创建缩放系数
            self.prompt_scale_seg = nn.ParameterList([
                nn.Parameter(torch.zeros(1)) for _ in range(4)
            ])
            self.prompt_scale_cls = nn.Parameter(torch.zeros(1))
        else:
            self.dec_prompt_mlp = None
            self.prompt_proj_layers = None
            self.prompt_scale_seg = None
            self.prompt_scale_cls = None

        from .decoder_UPerhead import UPerNet
        self.seg_head = UPerNet()
        
        self.cls_head_2cls = nn.Linear(embed_dim, 2)
        self.cls_head_4cls = nn.Linear(embed_dim, 4)

    def forward(self, x):
        if self.prompt:
            image = x[0]
            position_prompt = x[1]
            task_prompt = x[2]
            type_prompt = x[3]
            nature_prompt = x[4]
            B = image.shape[0]
            image = image.squeeze(1).permute(0, 3, 1, 2)
            prompt_input = torch.cat(
                [position_prompt, task_prompt, type_prompt, nature_prompt], dim=1
            )  # (B, 15)
        else:
            image = x.squeeze(1).permute(0, 3, 1, 2)
            B = image.shape[0]
                
        feats_cls, feats_seg = self.backbone.forward(image)

        if self.prompt:
            # 分类分支 prompt
            prompt_cls = self.dec_prompt_mlp(prompt_input).view(B, -1, 1, 1)
            feats_cls = feats_cls.view(B, -1, 1, 1)
            feats_cls = feats_cls + self.prompt_scale_cls * prompt_cls

            # 分割分支 prompt（每层独立缩放）
            feats_seg_with_prompt = []
            for i, feat in enumerate(feats_seg):
                prompt_seg = self.prompt_proj_layers[i](prompt_input).view(B, -1, 1, 1)
                feats_seg_with_prompt.append(feat + self.prompt_scale_seg[i] * prompt_seg)
            feats_seg = feats_seg_with_prompt
                 
        seg_out = self.seg_head(image, feats_seg)

        pooled_feats = feats_cls.view(B, -1)
        cls_out_2cls = self.cls_head_2cls(pooled_feats)
        cls_out_4cls = self.cls_head_4cls(pooled_feats)
        
        return (seg_out, cls_out_2cls, cls_out_4cls)
