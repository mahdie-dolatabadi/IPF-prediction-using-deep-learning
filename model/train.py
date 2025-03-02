import torch
from transformers import ViTHybridModel
from torchvision import models  # Pretrained models
from utils import FeatureTransformer, DecisionStep
from configs import HyperParameters  # Hyperparameter configuration object

params = HyperParameters("slope_train_vit_simple")
# print(params.n_epochs)  # Example access
class TabCT(torch.nn.Module):
    def __init__(self, cnn, num_features=4, output_dim=20, 
                 n_shared=2, n_d=64, n_a=64, n_ind=2, n_steps=4, relax=1.2, vbs=128):
        super(TabCT, self).__init__()

        # If shared layers are defined, create them
        if n_shared > 0:
            self.shared = torch.nn.ModuleList()
            # First shared layer from num_features to 2*(n_d+n_a)
            self.shared.append(torch.nn.Linear(num_features, 2*(n_d + n_a)))
            for _ in range(n_shared - 1):
                # Additional shared layers with dimensions (n_d+n_a)
                self.shared.append(torch.nn.Linear(n_d + n_a, 2*(n_d + n_a)))
        else:
            self.shared = None
        
        # First feature transformer step
        self.first_step = FeatureTransformer(num_features, n_d + n_a, self.shared, n_ind)
        
        # Additional decision steps
        self.steps = torch.nn.ModuleList()
        for _ in range(n_steps - 1):
            # Decision steps apply transformation to feature representation
            self.steps.append(DecisionStep(num_features, n_d, n_a, self.shared, n_ind, relax, vbs))
        
        # Fully connected layer for tabular features to output_dim
        self.fc_tab = torch.nn.Linear(n_d, output_dim)
        
        # Batch normalization for tabular features
        self.bn = torch.nn.BatchNorm1d(num_features)
        
        # Save the dimensions of features (n_d)
        self.n_d = n_d

        # Number of tabular features
        self.n_tab = params.n_tab  # n tabular features

        self.con1 = ViTHybridModel.from_pretrained("google/vit-hybrid-base-bit-384")
        # change configuration
        self.input_channel_after_mul = 64
        self.input_size_after_mul = 192
        
        # Adjust ViT configuration to work with our input size and channels
        self.con1.config.num_channels = self.input_channel_after_mul
        self.con1.config.image_size = self.input_size_after_mul
        self.con1.embeddings.patch_embeddings.num_channels = self.input_channel_after_mul
        self.con1.embeddings.patch_embeddings.image_size = (self.input_size_after_mul , self.input_size_after_mul)

        self.con1.embeddings.patch_embeddings.backbone.bit.config.num_channels = self.input_channel_after_mul

        self.con1.embeddings.patch_embeddings.backbone.bit.embedder.num_channels = self.input_channel_after_mul

        # First CNN Layer
        self.conv = self.con1.embeddings.patch_embeddings.backbone.bit.embedder.convolution

        # First Mask Layer
        self.W = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty((64, 3, 7, 7)), mean=0, std=0.01))
        self.B = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(64), mean=0, std=0.01))
        self.mask = self.con1.embeddings.patch_embeddings.backbone.bit.embedder.convolution
        self.mask.weight = self.W
        self.mask.bias = self.B

        # Rest of Embeddings
        self.embeddings = self.con1.embeddings
        self.embeddings.patch_embeddings.backbone.bit.embedder.convolution = UnlearnableIdentityConvModel()

        self.ct_cnn = self.con1.encoder
        self.norm = self.con1.layernorm
        self.pooler = self.con1.pooler


                        # model

                self.con1.embeddings.patch_embeddings.num_channels = self.input_channel_after_mul
                self.con1.embeddings.patch_embeddings.image_size = (self.input_size_after_mul , self.input_size_after_mul)

                self.con1.embeddings.patch_embeddings.backbone.bit.config.num_channels = self.input_channel_after_mul

                self.con1.embeddings.patch_embeddings.backbone.bit.embedder.num_channels = self.input_channel_after_mul

                # First CNN Layer
                self.conv = self.con1.embeddings.patch_embeddings.backbone.bit.embedder.convolution
                # print("First CNN Layer")

                # First Mask Layer
                self.W = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty((64, 3, 7, 7)), mean=0, std=0.01))
                self.B = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(64), mean=0, std=0.01))
                self.mask = self.con1.embeddings.patch_embeddings.backbone.bit.embedder.convolution
                self.mask.weight = self.W
                self.mask.bias = self.B
                # print("First Mask Layer")

                # Rest of Embeddings
                self.embeddings = self.con1.embeddings
                self.embeddings.patch_embeddings.backbone.bit.embedder.convolution = UnlearnableIdentityConvModel()
                # print(self.embeddings)
                # print("Rest of Embeddings")

                # Encoder(ViT), layernorm, pooler layer
                self.ct_cnn = self.con1.encoder
                # print(self.ct_cnn)
                self.norm = self.con1.layernorm
                self.pooler = self.con1.pooler
                # print("Encoder(ViT), layernorm, pooler layer")
        



        # Secondary feature extractor (ResNet18)
        self.ct_cnn_s = models.resnet18(pretrained=True)
        self.conv_s = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.ct_cnn_s.conv1 = torch.nn.Identity()
        self.W_s = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.empty((64, 3, 7, 7)), mean=0, std=0.01))
        self.B_s = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(64), mean=0, std=0.01))
        self.mask_s = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.mask_s.weight = self.W_s
        self.mask_s.bias = self.B_s

        # Dropout layer for regularization
        self.dropout = torch.nn.Dropout(p=0.3)

        # Intermediate fully connected layer
        self.fc_inter = torch.nn.Linear(params.cnn_dim_s + self.n_tab + params.cnn_dim, params.fc_dim)
    

        # Final fully connected layer to predict output
        self.fc = torch.nn.Linear(params.fc_dim, 1)

    def forward(self, x_ct, x_tab, masks):
        """
        Forward pass of the TabCT model.
        x_ct: Input CT scan images
        x_tab: Input tabular features (e.g., clinical information)
        masks: Masks for attention mechanisms in the model
        """
        # Batch normalization for tabular features
        x_temp = self.bn(x_tab)
        
        # First feature transformation step
        x_a = self.first_step(x_temp)[:, self.n_d:]
        
        # Initialize loss and output tensors
        loss = torch.zeros(1).to(x_temp.device)
        out = torch.zeros(x_temp.size(0), self.n_d).to(x_temp.device)
        priors = torch.ones(x_temp.shape).to(x_temp.device)
        
        # Process through additional decision steps
        for step in self.steps:
            x_te, l = step(x_temp, x_a, priors)
            out += torch.nn.functional.relu(x_te[:, :self.n_d])
            x_a = x_te[:, self.n_d:]
            loss += l
        
        # Concatenate CT scan features and duplicate masks for processing
        x_ct = torch.cat((x_ct, torch.cat((x_ct, x_ct), 1)), 1)
        masks = torch.cat((masks, torch.cat((masks, masks), 1)), 1)
        
        # Pass the concatenated input through the first CNN layer
        feature_map = self.conv(x_ct)  # ViT model
        feature_map_s = self.conv_s(x_ct)  # CNN model

        # Apply attention mechanism using masks
        relevance_map_s = self.mask_s(masks)  # CNN mask
        relevance_map = self.mask(masks)  # ViT mask

        # Apply element-wise multiplication (attention) to the feature maps
        ct_att = torch.mul(feature_map, relevance_map)  # ViT
        ct_att_s = torch.mul(feature_map_s, relevance_map_s)  # CNN
        
        # Process CNN feature map
        ct_f_s = self.ct_cnn_s(ct_att_s)

        # Process ViT feature map through embeddings, encoder, normalization, and pooler
        ct_f = self.embeddings(ct_att)
        ct_f = self.ct_cnn(ct_f)
        ct_f = self.norm(ct_f['last_hidden_state'])  # ViT features
        ct_f = self.pooler(ct_f)

        # Concatenate both CNN and ViT outputs
        x = torch.cat((ct_f_s, self.fc_tab(out)), -1)  # Concatenated feature map
        x = torch.cat((ct_f, x), -1)  # Concatenate final feature map

        # Apply dropout for regularization
        x = self.dropout(x)

        # Final fully connected layers
        x = self.fc_inter(x)
        x = self.fc(x)
        
        return x, loss
