class TabCT(nn.Module):
    def __init__(self, cnn,num_features=4, feature_dim=40, output_dim=20, num_decision_steps=3,  # 2 boud
                    relaxation_factor=1, batch_momentum=0.1, epsilon=0.00001, vgg_npy_path = None
                    , n_shared=2, n_d=64, n_a=64, n_ind=2, n_steps=4,relax=1.2,vbs=128):
        super(TabCT, self).__init__()
        
        if n_shared>0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(num_features,2*(n_d+n_a)))
            for x in range(n_shared-1):
                self.shared.append(nn.Linear(n_d+n_a,2*(n_d+n_a)))
        else:
            self.shared=None
        self.first_step = FeatureTransformer(num_features,n_d+n_a,self.shared,n_ind) 
        self.steps = nn.ModuleList()
        for x in range(n_steps-1):
            self.steps.append(DecisionStep(num_features,n_d,n_a,self.shared,n_ind,relax,vbs))
        self.fc_tab = nn.Linear(n_d,output_dim)
        self.bn = nn.BatchNorm1d(num_features)
        self.n_d = n_d

        # CT features
        cnn_dict = {'vit_base_patch16lung0' :None, 'vit_b_16':None, 'vgg16':models.vgg16, 'resnet18': models.resnet18, 'resnet34': models.resnet34, 'resnet50': models.resnet50,
                   'resnet101': models.resnet101, 'resnet152': models.resnet152, 'resnext50': models.resnext50_32x4d,
                   'resnext101': models.resnext101_32x8d}
        '''
        fully conected 
        self.fullyconected = nn.Linear(5, 25)
        '''



      


        self.num_features = num_features
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        # self.virtual_batch_size = virtual_batch_size
        # self.num_classes = num_classes
        self.epsilon = epsilon
        # feature dim
        self.out_dict = {'vit_base_patch16lung0':768, 'vit_b_16': 768, 'resnet18': 512, 'resnet34': 512, 'resnet50': 2048, 'resnet101': 2048, 'resnet152': 2048,
                         'resnext50': 2048, 'resnext101': 2048, "efnb0": 1280, "efnb1": 1280, "efnb2": 1408, 
                          "efnb3": 1536, "efnb4": 1792, "efnb5": 2048, "efnb6": 2304, "efnb7": 2560, "vgg16": 512}
        
        self.n_tab = hyp.n_tab # n tabular features
        
        # efficient net b0 to b7
        
        if cnn in cnn_dict.keys(): # resnet or resnext or vgg16
            if cnn == 'vgg16':
                # load vgg16 model badan inja be khatere none noudan ye if bezar

                self.ct_cnn = cnn_dict[cnn](pretrained = True).features[2:]
                self.conv = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                self.W = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty((64, 3, 3, 3)), mean=0, std=0.01))
                self.B = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(64), mean=0, std=0.01))

            elif cnn == 'vit_b_16':


                # model
                self.con1 = ViTHybridModel.from_pretrained("google/vit-hybrid-base-bit-384")
                # print("model")          


                # change configuration 
                self.input_channel_after_mul = 64
                self.input_size_after_mul = 192

                self.con1.config.num_channels = self.input_channel_after_mul
                self.con1.config.image_size = self.input_size_after_mul 

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



            elif cnn == 'vit_base_patch16lung0':
                self.con1 = AutoModelForImageClassification.from_pretrained("DunnBC22/vit-base-patch16-224-in21k_lung_and_colon_cancer").vit
                self.conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=3, padding=80)
                self.conv = self.con1.embeddings.patch_embeddings.backbone.embedder# nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))# self.ct_cnn.conv_proj 
                self.ct_cnn = self.con1.encoder

                self.W = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty((768, 3, 3, 3)), mean=0, std=0.01))
                self.B = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(768), mean=0, std=0.01))
                self.mask = self.con1.embeddings
                self.mask.patch_embeddings.projection.weight = self.W
                self.mask.patch_embeddings.projection.bias = self.B

                self.norm = self.con1.layernorm
                self.pooler = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").pooler
            else:   
                self.ct_cnn = cnn_dict[cnn](pretrained = True)
                
                # make single channel
                self.conv = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                self.ct_cnn.conv1 = nn.Identity()
                self.W = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty((64, 3, 3, 3)), mean=0, std=0.01))
                self.B = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(64), mean=0, std=0.01))
                # self.ct_cnn.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                
                # remove the fc layer/ add a simple linear layer
                self.ct_cnn.fc = nn.Linear(self.out_dict[cnn], hyp.cnn_dim)   # mapped to 64 dimensions, Identity()
            
        else:
            raise ValueError("cnn not recognized")
        
        # second feature extractor
        self.ct_cnn_s = models.resnet18(pretrained = True)
        self.conv_s = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.ct_cnn_s.conv1 = nn.Identity()
        self.W_s = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty((64, 3, 7, 7)), mean=0, std=0.01))
        self.B_s = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(64), mean=0, std=0.01))
        self.mask_s = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.mask_s.weight = self.W_s
        self.mask_s.bias = self.B_s
        self.dropout = nn.Dropout(p=0.3)
        print("second feature extractor")
        
        self.fc_inter = nn.Linear(hyp.cnn_dim_s + self.n_tab + hyp.cnn_dim, hyp.fc_dim) 

        self.BN_fc_inter = nn.BatchNorm1d(hyp.fc_dim,
                                        momentum=self.batch_momentum)

        self.fc = nn.Linear(hyp.fc_dim, 1)


    def forward(self, x_ct, x_tab, masks):

        x_temp = self.bn(x_tab)
        x_a = self.first_step(x_temp)[:,self.n_d:]
        loss = torch.zeros(1).to(x_temp.device)
        out = torch.zeros(x_temp.size(0),self.n_d).to(x_temp.device)
        priors = torch.ones(x_temp.shape).to(x_temp.device)
        for step in self.steps:
            x_te,l = step(x_temp,x_a,priors)
            out += nn.functional.relu(x_te[:,:self.n_d])
            x_a = x_te[:,self.n_d:]
            loss += l
        # all_loss = []
        # self.all_loss.append(loss)
        # print("tabular finished")
        # 1 + 1 + 1
        x_ct = torch.cat((x_ct, torch.cat((x_ct, x_ct), 1)), 1)
        masks = torch.cat((masks, torch.cat((masks, masks), 1)), 1)
        # print("input concatenate")


        feature_map = self.conv(x_ct) # ViT
        feature_map_s = self.conv_s(x_ct) # CNN
        # print("first layer image")

        
        relevance_map_s = self.mask_s(masks) # CNN
        relevance_map = self.mask(masks)  #self.B # ViT
        # print("first layer mask")

        # multiple element-wise
        ct_att = torch.mul(feature_map, relevance_map) # ViT
        ct_att_s = torch.mul(feature_map_s, relevance_map_s) # CNN
        # print("multiply both")

        ct_f_s = self.ct_cnn_s(ct_att_s) # CNN 
        # print("rest of CNN")

        # ViT
        # print(ct_att.size())
        # print(ct_att.shape)
        ct_f = self.embeddings(ct_att)
        # print("embeddings")
        ct_f = self.ct_cnn(ct_f)
        # print("ct_cnn")
        # print(ct_f)
        # print(ct_f['last_hidden_state'].size())
        # print(type(ct_f))
        ct_f = self.norm(ct_f['last_hidden_state']) # ct features
        # print("norm")
        ct_f = self.pooler(ct_f)
        # print("pooler")
        # print("rest of ViT")

        # concatenate
        x = torch.cat((ct_f_s, self.fc_tab(out)), -1) # concat on last axis output_aggregated  # changed
        x = torch.cat((ct_f, x), -1) # concat on last axis #changed
        # print("concatenate outputs")

        # dropout
        x = self.dropout(x)

        x = self.fc_inter(x)

        x = self.fc(x)
        
        return x, loss