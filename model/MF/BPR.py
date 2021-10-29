from model import basemodel, loss_func, scorer
from ann import sampler
import torch
class BPRRecommender(basemodel.UserItemIDTowerRecommender):

    def get_fields(self):
        return {'user_id', 'item_id', 'rating'}

    def build_user_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_users, self.embed_dim)
    
    def config_loss(self):
        return loss_func.BPRLoss()
    
    def config_scorer(self):
        return scorer.InnerProductScorer()

    def build_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items-1, self.score_func)