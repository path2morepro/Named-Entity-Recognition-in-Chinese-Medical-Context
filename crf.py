import torch
import torch.nn as nn
from typing import List, Optional


class CRF(nn.Module):
    def __init__(self, num_tags: int, batch_first: bool = False):
        if num_tags <= 0:
            raise ValueError('num_tags 一定要是大于0的自然数！！！')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first

        # 转移概率矩阵
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        # 初始化转移矩阵
        self.reset_parameters()

    def reset_parameters(self):
        '''
        初始化转移概率矩阵的参数，满足 (-0.1, 0.1) 的均匀分布
        '''
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'sum',
    ):
        '''
        前向传播，计算损失
        :param emissions: 当 batch_first=True，(batch_size, seq_length, num_tags)
        :param tags: 标签，(batch_size, seq_length)
        :param mask: 掩码，(batch_size, seq_length)
        :param reduction: 返回损失的格式：'sum', 'none', 'mean'
        :return:
        '''
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None):
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _compute_score(
            self, emissions: torch.Tensor, tags: torch.LongTensor,
            mask: torch.ByteTensor):
        '''
        计算带标签的真实语句的分数值
        :param emissions: (seq_length, batch_size, num_tags)
        :param tags: (seq_length, batch_size)
        :param mask: (seq_length, batch_size)
        :return:
        '''
        seq_length, batch_size = tags.shape
        mask = mask.float()

        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor):
        '''
        计算句子前向传播分数
        :param emissions: (seq_length, batch_size, num_tags)
        :param mask: (seq_length, batch_size)
        :return:
        '''
        seq_length = emissions.size(0)

        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor):
        # 维特比解码
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        for i in range(1, seq_length):
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list
