from dataclasses import dataclass
import torch

@dataclass
class RankingLoss:
    
    margin: float = 0
    gold_margin: float = 0
    gold_weight: float=1
    no_gold:bool=False
    no_cand:bool=False

    def __call__(self, score,summary_score):
        ones = torch.ones_like(score)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        TotalLoss = loss_func(score, score, ones)
        # candidate loss
        n = score.size(1)
        if not self.no_cand:
            for i in range(1, n):
                pos_score = score[:, :-i]
                neg_score = score[:, i:]
                pos_score = pos_score.contiguous().view(-1)
                neg_score = neg_score.contiguous().view(-1)
                ones = torch.ones_like(pos_score)
                loss_func = torch.nn.MarginRankingLoss(self.margin * i)
                loss = loss_func(pos_score, neg_score, ones)
                TotalLoss += loss
        if self.no_gold:
            return TotalLoss
        # gold summary loss
        pos_score = summary_score.unsqueeze(-1).expand_as(score)
        neg_score = score
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        ones = torch.ones_like(pos_score)
        loss_func = torch.nn.MarginRankingLoss(self.gold_margin)
        TotalLoss += self.gold_weight * loss_func(pos_score, neg_score, ones)
        return TotalLoss