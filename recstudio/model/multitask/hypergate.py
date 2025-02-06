import torch
import torch.nn as nn
from collections import defaultdict
from recstudio.model.multitask._base import _MultiTaskBase
from ..module import ctr, MLPModule, get_act

class MLP(torch.nn.Module):
    def __init__(self, mlp_layers, activation_func='ReLU', dropout=0.0, bias=True, batch_norm=False, last_activation=True, last_bn=True):
        super().__init__()
        self.mlp_layers = mlp_layers
        self.batch_norm = batch_norm
        self.bias = bias
        self.dropout = dropout
        self.activation_func = activation_func
        self.model = []
        last_bn = self.batch_norm and last_bn
        for idx, layer in enumerate((zip(self.mlp_layers[: -1], self.mlp_layers[1:]))):
            self.model.append(torch.nn.Dropout(dropout))
            self.model.append(torch.nn.Linear(*layer, bias=bias))
            if (idx == len(mlp_layers)-2 and last_bn) or (idx < len(mlp_layers)-2 and batch_norm):
                self.model.append(torch.nn.BatchNorm1d(layer[-1]))
            if ( (idx == len(mlp_layers)-2 and last_activation and activation_func is not None)
                or (idx < len(mlp_layers)-2 and activation_func is not None) ):
                activation = get_act(activation_func, dim=layer[-1])
                self.model.append(activation)
        
        self.model = nn.ModuleList(self.model)

    def add_modules(self, *args):
        """
        Adds modules into the MLP model after obtaining the instance.

        Args:
            args(variadic argument): the modules to be added into MLP model.
        """
        for block in args:
            assert isinstance(block, torch.nn.Module)

        for block in args:
            self.model.add_module(str(len(self.model._modules)), block)

    def forward(self, inputs, bias_list):
        x = inputs
        linear_cnt = 0
        for model in self.model:
            x = model(x)
            if isinstance(model, torch.nn.Linear):
                x = x + bias_list[linear_cnt]
                linear_cnt += 1
        return x


class HyperGate(_MultiTaskBase):

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        model_config = self.config['model']
        self.embedding = ctr.Embeddings(
            self.fields, self.embed_dim, train_data, 
            # share_dense_embedding=False, share_dense_embed_dim=2*self.embed_dim
        )
        assert isinstance(self.frating, list), f'Expect rating_field to be a list, but got {self.frating}.'
        enable_domain_gate = model_config.get('enable_embedding_gate', False)
        enable_biasing_gate = model_config.get('enable_biasing_gate', False)
        enable_fusion_gate = model_config.get('enable_fusion_gate', False)

        gate_input_type = model_config.get('gate_input_type', 'augmented_embedding')

        self.enable_domain_gate = enable_domain_gate
        self.enable_biasing_gate = enable_biasing_gate
        self.enable_fusion_gate = enable_fusion_gate
        self.gate_input_type = gate_input_type

        self.domain_aug_module = None
        self.task_aug_module = None
        self.domain_anchor_embedding = None
        self.task_anchor_embedding = None

        if gate_input_type == 'augmented_embedding':
            self._init_dt_embedding()
            self._init_augmentation_modules()
            domain_control_dim = model_config['aug_embed_dim'] * 2
            task_control_dim = model_config['aug_embed_dim'] * 2
        elif gate_input_type == 'id':
            self._init_dt_embedding()
            domain_control_dim = model_config['aug_embed_dim']
            task_control_dim = model_config['aug_embed_dim']
        else:
            domain_control_dim = self.embedding.output_dim
            task_control_dim = 0

        if enable_domain_gate:
            self.domain_gate = ctr.SENet(
                c_in_dim=domain_control_dim, 
                x_in_dim=self.embedding.output_dim,
            )
            if gate_input_type in {"augmented_embedding", "id"}:
                self.embedding = ctr.Embeddings(
                    self.fields - set([self.config['eval'].get('multi_domain_key')]), self.embed_dim, train_data, 
                )
                self.domain_embedding = nn.Embedding(len(self.domain_dict), self.embed_dim, padding_idx=0)
                self.embedding.output_dim += self.embed_dim
            else:
                pass
        else:
            self.domain_gate = None

        if enable_biasing_gate:
            self.biasing_gate = nn.ModuleList([
                MLPModule(
                    [domain_control_dim, domain_control_dim, expert_hidden],
                    last_activation=False,
                )
                for expert_hidden in model_config['expert_mlp_layer']
            ])

            self.experts = nn.ModuleList([
                MLP(
                    [self.embedding.output_dim] + model_config['expert_mlp_layer'],
                    model_config['expert_activation'], 
                    model_config['expert_dropout'],
                    batch_norm=model_config['expert_batch_norm']
                )
                for _ in range(model_config['num_experts'])
            ])
        else:
            self.biasing_gate = None
            self.experts = nn.ModuleList([
                MLPModule(
                    [self.embedding.output_dim] + model_config['expert_mlp_layer'],
                    model_config['expert_activation'], 
                    model_config['expert_dropout'],
                    batch_norm=model_config['expert_batch_norm'])
                for _ in range(model_config['num_experts'])
            ])

        if enable_fusion_gate:
            fg_control_dim = domain_control_dim + task_control_dim
        else:
            fg_control_dim = self.embedding.output_dim

        self.fusion_gates = nn.ModuleDict({
            r: MLPModule(
                [fg_control_dim] + model_config['gate_mlp_layer'] + [model_config['num_experts']],
                model_config['gate_activation'], 
                model_config['gate_dropout'],
                batch_norm=model_config['gate_batch_norm'],
                last_activation=False
            )
            for r in self.frating
        })
        for _, g in self.fusion_gates.items():
            g.add_modules(nn.Softmax(-1))

        self.towers = nn.ModuleDict({
            r: MLPModule(
                [model_config['expert_mlp_layer'][-1]] + model_config['tower_mlp_layer'] + [1],
                model_config['tower_activation'], 
                model_config['tower_dropout'],
                batch_norm=model_config['tower_batch_norm'],
                last_activation=False, 
                last_bn=False)
            for r in self.frating
        })
        
    def _init_dt_embedding(self):
        model_config = self.config['model']
        aug_dim = model_config['aug_embed_dim']
        self.domain_anchor_embedding = nn.Embedding(len(self.domain_dict), aug_dim, padding_idx=0)
        self.task_anchor_embedding = nn.Embedding(len(self.frating), aug_dim)
    
    def _init_augmentation_modules(self):
        model_config = self.config['model']
        aug_dim = model_config['aug_embed_dim']
        _layers = [self.embedding.output_dim-self.embed_dim] + model_config['aug_layers'] + [aug_dim]
        self.domain_aug_module = MLPModule(
            mlp_layers=_layers,
            activation_func='relu',
            dropout=model_config['aug_dropout'],
            last_activation=False,
            last_bn=False
        )
        self.task_aug_module = MLPModule(
            mlp_layers=_layers,
            activation_func='relu',
            dropout=model_config['aug_dropout'],
            last_activation=False,
            last_bn=False
        )


    def score(self, batch):
        emb = self.embedding(batch).flatten(1)
        domain_key = self.config['eval'].get('multi_domain_key')
        domain_id = batch[domain_key]
        aux_loss = 0.0
        lambda_d, lambda_t = self.config['model']['lambda_d'], self.config['model']['lambda_t']

        if self.gate_input_type == 'augmented_embedding':
            if self.enable_domain_gate or self.enable_biasing_gate:
                domain_aug = self.domain_aug_module(emb.detach())
                domain_anchor_emb = self.domain_anchor_embedding(domain_id)
                domain_control = torch.cat([domain_aug, domain_anchor_emb.detach()], dim=-1)
                # domain_control = domain_aug
                _scores = domain_aug @ self.domain_anchor_embedding.weight[1:].T
                d_loss = nn.CrossEntropyLoss(reduction='mean')(_scores, domain_id-1)
                aux_loss = aux_loss + d_loss * lambda_d

            # task
            if self.enable_fusion_gate:
                task_aug = self.task_aug_module(emb.detach())    #[B, D]
                task_anchor_emb = self.task_anchor_embedding.weight # [T, D]
                label = {r: batch[r] for r in self.frating}

                # loss 1:
                task_score = task_aug @ task_anchor_emb.T   # [B, T]
                t_loss_dict = {r: nn.BCEWithLogitsLoss(reduction='mean')(task_score[:, i], label[r].float()) for i, r in enumerate(self.frating)}
                t_loss = sum(t_loss_dict.values())

                # loss 2:
                # avg_task_emb = torch.stack([task_aug[label[r]==1].mean(dim=0) for r in self.frating], dim=0)
                # task_score = avg_task_emb @ task_anchor_emb.T
                # t_loss = nn.CrossEntropyLoss(reduction='mean')(task_score, torch.arange(len(self.frating), device=task_score.device))
            
                aux_loss = aux_loss + t_loss * lambda_t

        elif self.gate_input_type == 'id':
            domain_control = self.domain_anchor_embedding(domain_id)
            task_anchor_emb = self.task_anchor_embedding.weight  # [T, D]
        else:
            domain_control = emb

        if self.enable_domain_gate:
            if self.gate_input_type in {'augmented_embedding', 'id'}:
                domain_emb = self.domain_embedding(domain_id)
                emb = torch.cat([emb, domain_emb], dim=-1)
                # if self.gate_input_type == 'augmented_embedding':
                #     domain_control = torch.cat([emb, domain_control], dim=-1)
            emb, _ = self.domain_gate(domain_control, emb)

        if self.enable_biasing_gate:
            bias = [gate(domain_control) for gate in self.biasing_gate]  # B x L
            experts_out = torch.stack([e(emb, bias) for e in self.experts], dim=1)        # B x E x De
        else:
            experts_out = torch.stack([e(emb) for e in self.experts], dim=1)        # B x E x De

        score = defaultdict(dict)
        n_task = 0
        for r, gate in self.fusion_gates.items():
            if not self.enable_fusion_gate or (self.gate_input_type not in {'augmented_embedding', 'id'}):
                task_control = emb
            elif self.gate_input_type == 'id':
                anchor_emb = task_anchor_emb[n_task].unsqueeze(0).expand(emb.shape[0], -1)
                task_control = torch.cat([anchor_emb, domain_control], dim=-1)
            else:
                anchor_emb = task_anchor_emb[n_task].unsqueeze(0).expand(emb.shape[0], -1)
                task_control = torch.cat([task_aug, anchor_emb.detach(), domain_control], dim=-1)
                # task_control = torch.cat([emb, domain_control, domain_anchor_emb], dim=-1)
            
            gate_out = gate(task_control)                            # B x E
            mmoe_out = (gate_out.unsqueeze(-1) * experts_out).sum(1)            # B x De
            yh = self.towers[r](mmoe_out).squeeze(-1)
            score[r]['score'] = yh
            n_task += 1

        score['cl_loss'] = aux_loss
        return score


    def training_step(self, batch):
        y_h, output = self.forward(batch)
        weights = self.config['train'].get('multitask_weights', None)
        if weights is None:
            weights = [1.0] * len(self.frating)
        assert len(weights) == len(self.frating), \
            f'Expect {len(self.frating)} float(s) for weights, but got {self.config["train"]["weights"]} with length {len(weights)}.'
        weights = torch.tensor(weights, device=self.device)

        loss = {}
        loss['loss'] = 0.0
        for i, r in enumerate(self.frating):
            loss[r] = self.loss_fn(**y_h[r])
            loss['loss'] = loss['loss'] + weights[i] * loss[r]

        if 'cl_loss' in output:
            cl_loss = output['cl_loss']
            loss['loss'] = loss['loss'] + cl_loss
            # print(f'aux_loss: {cl_loss.item()}')
        return loss