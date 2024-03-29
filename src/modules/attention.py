import torch
import torch.nn as nn
# from torch.autograd import Variable
import src.utils.init as my_init
from .basic import BottleSoftmax

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = d_model ** 0.5
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = BottleSoftmax(dim=1)

    def forward(self, q, k, v, attn_mask=None):
        """
        :type attn_mask: torch.FloatTensor
        :param attn_mask: Mask of the attention.
            3D tensor with shape [batch_size, time_step_key, time_step_value]
        """
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                    'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{}.'.format(attn_mask.size(), attn.size())
            attn = attn.masked_fill(attn_mask, -1e18)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class BahdanauAttention(nn.Module):

    def __init__(self, query_size, key_size, hidden_size=None, cover_size=None):
        super().__init__()

        self.query_size = query_size
        self.key_size = key_size
        self.cover_size=cover_size

        if hidden_size is None:
            hidden_size = key_size

        self.hidden_size = hidden_size

        self.linear_key = nn.Linear(in_features=self.key_size, out_features=self.hidden_size)
        self.linear_query = nn.Linear(in_features=self.query_size, out_features=self.hidden_size)

        if cover_size is not None:
            self.linear_cov = nn.Linear(in_features=self.cover_size, out_features=self.hidden_size)

            self.linear_cov_cov = nn.Linear(in_features=self.cover_size, out_features=self.hidden_size)
            self.linear_key_cov = nn.Linear(in_features=self.key_size, out_features=self.hidden_size)
            self.linear_query_cov = nn.Linear(in_features=self.query_size, out_features=self.hidden_size)
            self.linear_att_cov = nn.Linear(in_features=1, out_features=self.hidden_size)
            self.gru = nn.GRUCell(input_size=hidden_size, hidden_size=cover_size)

        self.linear_logit = nn.Linear(in_features=self.hidden_size, out_features=1)

        self.softmax = BottleSoftmax(dim=1)
        self.tanh = nn.Tanh()

        self._reset_parameters()

    def _reset_parameters(self):
        for weight in self.parameters():
            my_init.default_init(weight)
        if self.cover_size is not None:
            for weight in self.gru.parameters():
                my_init.rnn_init(weight)

    def compute_cache(self, memory):

        return self.linear_key(memory)

    def forward(self, query, memory, cache=None, mask=None, coverage=None):
        """
        :param query: Key tensor.
            with shape [batch_size, input_size]

        :param memory: Memory tensor.
            with shape [batch_size, mem_len, input_size]

        :param mask: Memory mask which the PAD position is marked with true.
            with shape [batch_size, mem_len]

        :param coverage: Coverage vector
            with shape [batch_size, mem_len, dim_cover]
        """

        if query.dim() == 2:
            query = query.unsqueeze(1)   # [batch_size, 1, input_size]
            one_step = True
        else:
            one_step = False

        batch_size, q_len, q_size = query.size()
        _, m_len, m_size = memory.size()

        q = self.linear_query(query.view(-1, q_size))  # [batch_size, q_len, hidden_size]

        if cache is not None:
            k = cache
        else:
            k = self.linear_key(memory.view(-1, m_size))  # [batch_size, m_len, hidden_size]

        # logit = q.unsqueeze(0) + k # [mem_len, batch_size, dim]
        logits = q.view(batch_size, q_len, 1, -1) + k.view(batch_size, 1, m_len, -1)

        if coverage is not None:
            bt_size, cover_len, cover_size = coverage.size()
            assert cover_len == m_len, 'The length of coverage vector dose not equal memory.'
            cover = self.linear_cov(coverage.view(-1, cover_size))
            logits += cover.view(batch_size, 1, cover_len, -1)

            cover_u = self.linear_cov_cov(coverage.view(-1, cover_size))
            q_u = self.linear_query_cov(query.view(-1, q_size))
            k_u = self.linear_key_cov(memory.view(-1, m_size))

            cover_update = q_u.view(batch_size, q_len, 1, -1) + k_u.view(batch_size, 1, m_len, -1) + \
                cover_u.view(batch_size, 1, cover_len, -1)

        logits = self.tanh(logits)
        logits = self.linear_logit(logits.view(-1, self.hidden_size)).view(batch_size, q_len, m_len)

        if mask is not None:
            mask_ = mask.unsqueeze(1)  # [batch_size, 1, m_len]
            logits = logits.masked_fill(mask_, -1e18)

        weights = self.softmax(logits)  # [batch_size, q_len, m_len]

        if coverage is not None:
            weights_u = self.linear_att_cov(weights.unsqueeze(3))
            cover_update += weights_u
            cover_update = cover_update.view(batch_size*m_len, -1)  # [batch_size*m_len, hidden_size]
            coverage = self.gru(cover_update, coverage.view(batch_size*m_len, -1)).view(batch_size, m_len, -1)


        # [batch_size, q_len, m_len] @ [batch_size, m_len, m_size]
        # ==> [batch_size, q_len, m_size]
        attns = torch.bmm(weights, memory)

        if one_step:
            attns = attns.squeeze(1)  # ==> [batch_size, q_len]

        return attns, weights, coverage
