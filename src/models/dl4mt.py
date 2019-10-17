import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.common_utils import Vocab
import src.utils.init as my_init
from src.modules.embeddings import Embeddings
from src.modules.cgru import CGRUCell
from src.modules.rnn import RNN
from src.utils.beam_search import tile_batch, tensor_gather_helper, mask_scores


class Encoder(nn.Module):
    def __init__(self,
                 n_words,
                 input_size,
                 hidden_size
                 ):

        super(Encoder, self).__init__()

        # Use PAD
        self.embedding = Embeddings(num_embeddings=n_words,
                                    embedding_dim=input_size,
                                    dropout=0.0,
                                    add_position_embedding=False)

        self.gru = RNN(type="gru", batch_first=True, input_size=input_size, hidden_size=hidden_size,
                       bidirectional=True)

    def forward(self, x):
        """
        :param x: Input sequence.
            with shape [batch_size, seq_len, input_size]
        """
        x_mask = x.detach().eq(Vocab.PAD)

        emb = self.embedding(x)

        ctx, _ = self.gru(emb, x_mask)

        return ctx, x_mask


class Decoder(nn.Module):

    def __init__(self,
                 n_words,
                 input_size,
                 hidden_size,
                 bridge_type="mlp",
                 dropout_rate=0.0,
                 cover_size=None,
                 copy_attn=False):

        super(Decoder, self).__init__()

        self.bridge_type = bridge_type
        self.hidden_size = hidden_size
        self.context_size = hidden_size * 2

        self.embedding = Embeddings(num_embeddings=n_words,
                                    embedding_dim=input_size,
                                    dropout=0.0,
                                    add_position_embedding=False)

        self.cgru_cell = CGRUCell(input_size=input_size, hidden_size=hidden_size, cover_size=cover_size)

        self.linear_input = nn.Linear(in_features=input_size, out_features=input_size)
        self.linear_hidden = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.linear_ctx = nn.Linear(in_features=hidden_size * 2, out_features=input_size)

        if copy_attn:
            self.linear_input_copy = nn.Linear(in_features=input_size, out_features=1)
            self.linear_hidden_copy = nn.Linear(in_features=hidden_size, out_features=1)
            self.linear_ctx_copy = nn.Linear(in_features=hidden_size * 2, out_features=1)
            self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout_rate)

        self._reset_parameters()

        self._build_bridge()

        self.tanh = nn.Tanh()

    def _reset_parameters(self):

        my_init.default_init(self.linear_input.weight)
        my_init.default_init(self.linear_hidden.weight)
        my_init.default_init(self.linear_ctx.weight)

    def _build_bridge(self):

        if self.bridge_type == "mlp":
            self.linear_bridge = nn.Linear(in_features=self.context_size, out_features=self.hidden_size)
            my_init.default_init(self.linear_bridge.weight)
        elif self.bridge_type == "zero":
            pass
        else:
            raise ValueError("Unknown bridge type {0}".format(self.bridge_type))

    def init_decoder(self, context, mask):

        # Generate init hidden
        if self.bridge_type == "mlp":

            no_pad_mask = 1.0 - mask.float()
            ctx_mean = (context * no_pad_mask.unsqueeze(2)).sum(1) / no_pad_mask.unsqueeze(2).sum(1)
            # dec_init = F.tanh(self.linear_bridge(ctx_mean))
            dec_init = self.tanh(self.linear_bridge(ctx_mean))

        elif self.bridge_type == "zero":
            batch_size = context.size(0)
            dec_init = context.new(batch_size, self.hidden_size).zero_()
        else:
            raise ValueError("Unknown bridge type {0}".format(self.bridge_type))

        dec_cache = self.cgru_cell.compute_cache(context)

        return dec_init, dec_cache

    def init_coverage(self, context, cover_size):

        batch_size = context.size(0)
        src_len = context.size(1)
        cover_init = context.new(batch_size, src_len, cover_size).zero_()

        return cover_init

    def forward(self, y, context, context_mask, hidden, one_step=False, cache=None, coverage=None, copy_attn=False):

        emb = self.embedding(y)  # [seq_len, batch_size, dim]

        if one_step:
            (out, attn), hidden, coverage, attn_w = self.cgru_cell(emb, hidden, context, context_mask, cache, coverage)
            attn_w = attn_w.squeeze(1)
        else:
            # emb: [seq_len, batch_size, dim]
            out = []
            attn = []
            attn_w = []

            for emb_t in torch.split(emb, split_size_or_sections=1, dim=0):

                (out_t, attn_t), hidden, coverage, attn_w_t = self.cgru_cell(emb_t.squeeze(0), hidden,
                                                                             context, context_mask, cache, coverage)
                out += [out_t]
                attn += [attn_t]
                attn_w += [attn_w_t.squeeze(1)]

            out = torch.stack(out)     # h_tgt
            attn = torch.stack(attn)   # attention value: a * h_src
            if copy_attn:
                attn_w = torch.stack(attn_w)

        logits = self.linear_input(emb) + self.linear_hidden(out) + self.linear_ctx(attn)

        logits_copy = None
        if copy_attn:
            logits_copy = self.linear_input_copy(emb) + self.linear_hidden_copy(out) + self.linear_ctx_copy(attn)
            logits_copy = self.sigmoid(logits_copy)  # [seq_len, batch_size, 1]

        # logits = F.tanh(logits)
        logits = self.tanh(logits)

        logits = self.dropout(logits)  # [seq_len, batch_size, dim]

        return logits, hidden, coverage, attn_w, logits_copy


class Generator(nn.Module):

    def __init__(self, n_words, hidden_size, shared_weight=None, padding_idx=-1):

        super(Generator, self).__init__()

        self.n_words = n_words
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

        self.proj = nn.Linear(self.hidden_size, self.n_words, bias=False)
        self.actn = nn.LogSoftmax(dim=-1)

        if shared_weight is not None:
            self.proj.weight = shared_weight
        else:
            self._reset_parameters()

    def _reset_parameters(self):

        my_init.embedding_init(self.proj.weight)

    def forward(self, input):
        """
        input == > Linear == > LogSoftmax
        """
        return self.actn(self.proj(input))


class CopyGenerator(nn.Module):

    def __init__(self, n_words, hidden_size, shared_weight=None, padding_idx=-1):

        super(CopyGenerator, self).__init__()

        self.n_words = n_words
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

        self.proj = nn.Linear(self.hidden_size, self.n_words, bias=False)
        self.actn = nn.Softmax(dim=-1)

        if shared_weight is not None:
            self.proj.weight = shared_weight
        else:
            self._reset_parameters()

    def _reset_parameters(self):

        my_init.embedding_init(self.proj.weight)

    def forward(self, input, p_gen, attn_w):
        """
        input == > Linear == > Softmax
        attn_w: [batch * len_tgt, len_src]
        p_gen: [batch * len_tgt, 1]
        """
        # [batch * len_tgt, vocab_tgt]
        p_vocab = self.actn(self.proj(input))
        p_vocab = torch.mul(p_vocab, p_gen.expand_as(p_vocab))
        p_copy = torch.mul(attn_w, 1 - p_gen.expand_as(attn_w))
        # p_vocab = p_vocab * p_gen
        # p_copy = (1 - p_gen) * attn_w

        return p_vocab, p_copy


class DL4MT(nn.Module):

    def __init__(self, n_src_vocab, n_tgt_vocab, d_word_vec, d_model, dropout,
                 proj_share_weight, bridge_type="mlp", cover_size=None, copy_attn=False, **kwargs):

        super().__init__()

        self.encoder = Encoder(n_words=n_src_vocab, input_size=d_word_vec, hidden_size=d_model)

        self.decoder = Decoder(n_words=n_tgt_vocab, input_size=d_word_vec, hidden_size=d_model, dropout_rate=dropout,
                               bridge_type=bridge_type, cover_size=cover_size, copy_attn=copy_attn)

        if copy_attn:
            if proj_share_weight is False:
                generator = CopyGenerator(n_words=n_tgt_vocab, hidden_size=d_word_vec, padding_idx=Vocab.PAD)
            else:
                generator = CopyGenerator(n_words=n_tgt_vocab, hidden_size=d_word_vec, padding_idx=Vocab.PAD,
                                          shared_weight=self.decoder.embedding.embeddings.weight
                                          )
        else:
            if proj_share_weight is False:
                generator = Generator(n_words=n_tgt_vocab, hidden_size=d_word_vec, padding_idx=Vocab.PAD)
            else:
                generator = Generator(n_words=n_tgt_vocab, hidden_size=d_word_vec, padding_idx=Vocab.PAD,
                                      shared_weight=self.decoder.embedding.embeddings.weight
                                     )

        self.generator = generator
        self.cover_size = cover_size
        self.copy_attn = copy_attn

    def force_teaching(self, x, y):

        ctx, ctx_mask = self.encoder(x)

        dec_init, dec_cache = self.decoder.init_decoder(ctx, ctx_mask)

        coverage_init = None
        if self.cover_size is not None:
            coverage_init = self.decoder.init_coverage(ctx, self.cover_size)

        logits, _, __, attn_w, logits_copy = self.decoder(y,
                                                          context=ctx,
                                                          context_mask=ctx_mask,
                                                          one_step=False,
                                                          hidden=dec_init,
                                                          cache=dec_cache,
                                                          coverage=coverage_init,
                                                          copy_attn=self.copy_attn)  # [tgt_len, batch_size, dim]

        # Convert to batch-first mode.
        if self.copy_attn:
            return logits.transpose(1, 0).contiguous(), attn_w, logits_copy.transpose(1, 0).contiguous()
        return logits.transpose(1, 0).contiguous(), None, None

    def batch_beam_search(self, x, x_ext=None, beam_size=5, max_steps=150, copy_attn=False, n_words_ext=None):

        if copy_attn:
            assert x_ext is not None, 'if use copy_attn, then x_ext is needed.'

        batch_size = x.size(0)
        src_len = x.size(1)

        ctx, ctx_mask = self.encoder(x)
        dec_init, dec_cache = self.decoder.init_decoder(ctx, ctx_mask)

        ctx = tile_batch(ctx, multiplier=beam_size, batch_dim=0)
        dec_cache = tile_batch(dec_cache, multiplier=beam_size, batch_dim=0)
        hiddens = tile_batch(dec_init, multiplier=beam_size, batch_dim=0)
        ctx_mask = tile_batch(ctx_mask, multiplier=beam_size, batch_dim=0)

        covers = None
        if self.cover_size is not None:
            cov_init = ctx.new(batch_size, src_len, self.cover_size).zero_()
            covers = tile_batch(cov_init, multiplier=beam_size, batch_dim=0)

        beam_mask = ctx_mask.new(batch_size, beam_size).fill_(1).float()
        dec_memory_len = ctx_mask.new(batch_size, beam_size).zero_().float()
        beam_scores = ctx_mask.new(batch_size, beam_size).zero_().float()
        final_word_indices = x.new(batch_size, beam_size, 1).fill_(Vocab.BOS)

        if copy_attn:
            final_word_indices_copy = x.new(batch_size, beam_size, 1).fill_(Vocab.BOS)

        for t in range(max_steps):

            if copy_attn:
                y_indices = final_word_indices_copy
            else:
                y_indices = final_word_indices
            logits, hiddens, covers, attn_w, logits_copy = \
                self.decoder(y=y_indices[:, :, -1].contiguous().view(batch_size * beam_size, ),
                             hidden=hiddens.view(batch_size * beam_size, -1),
                             context=ctx,
                             context_mask=ctx_mask,
                             one_step=True,
                             cache=dec_cache,
                             coverage=covers,
                             copy_attn=copy_attn,
                             )

            hiddens = hiddens.view(batch_size, beam_size, -1)

            if self.cover_size is not None:
                covers = covers.view(batch_size, beam_size, src_len, -1)

            # if use copy need to modify
            if copy_attn:
                scores, copy_scores = self.generator(logits, logits_copy, attn_w)
                vocab_tgt_size = scores.size(-1)
                if n_words_ext > vocab_tgt_size:
                    expand = scores.new(scores.size(0), n_words_ext - scores.size(-1)).zero_()
                    scores = torch.cat((scores, expand), 1)  # [B * Bm, num_tokens]

                x_ext_t = x_ext.unsqueeze(1)  # [B, 1, len_src]
                x_ext_t = x_ext_t.repeat(1, beam_size, 1)  # [B, Bm, len_src]
                x_ext_t = x_ext_t.view(-1, x_ext_t.size(2))  # [B * Bm, len_src]
                scores.scatter_add_(1, x_ext_t, copy_scores)
                # mask_ss = scores.eq(0.).float()
                # scores = scores + mask_ss
                next_scores = - torch.log(scores)        # [B * Bm, num_tokens]
            else:
                next_scores = - self.generator(logits)  # [B * Bm, N]

            next_scores = next_scores.view(batch_size, beam_size, -1)
            next_scores = mask_scores(next_scores, beam_mask=beam_mask)

            beam_scores = next_scores + beam_scores.unsqueeze(2)  # [B, Bm, N] + [B, Bm, 1] ==> [B, Bm, N]

            vocab_size = beam_scores.size(-1)
            if t == 0:
                beam_scores = beam_scores[:, 0, :].contiguous()   # [B, 1, N] first step only BOS

            beam_scores = beam_scores.view(batch_size, -1)

            # Get topK smallest with beams
            beam_scores, indices = torch.topk(beam_scores, k=beam_size, dim=-1, largest=False, sorted=False)
            next_beam_ids = torch.div(indices, vocab_size)
            next_word_ids = indices % vocab_size

            if copy_attn:
                next_word_ids_copy = indices % vocab_size

            # gather beam cache
            dec_memory_len = tensor_gather_helper(gather_indices=next_beam_ids,
                                                  gather_from=dec_memory_len,
                                                  batch_size=batch_size,
                                                  beam_size=beam_size,
                                                  gather_shape=[-1])

            hiddens = tensor_gather_helper(gather_indices=next_beam_ids,
                                           gather_from=hiddens,
                                           batch_size=batch_size,
                                           beam_size=beam_size,
                                           gather_shape=[batch_size * beam_size, -1])

            if self.cover_size is not None:
                covers = tensor_gather_helper(gather_indices=next_beam_ids,
                                              gather_from=covers,
                                              batch_size=batch_size,
                                              beam_size=beam_size,
                                              gather_shape=[batch_size * beam_size, src_len, -1])
                covers = covers.view(batch_size * beam_size, src_len, -1)

            beam_mask = tensor_gather_helper(gather_indices=next_beam_ids,
                                             gather_from=beam_mask,
                                             batch_size=batch_size,
                                             beam_size=beam_size,
                                             gather_shape=[-1])

            final_word_indices = tensor_gather_helper(gather_indices=next_beam_ids,
                                                      gather_from=final_word_indices,
                                                      batch_size=batch_size,
                                                      beam_size=beam_size,
                                                      gather_shape=[batch_size * beam_size, -1])
            if copy_attn:
                final_word_indices_copy = tensor_gather_helper(gather_indices=next_beam_ids,
                                                               gather_from=final_word_indices_copy,
                                                               batch_size=batch_size,
                                                               beam_size=beam_size,
                                                               gather_shape=[batch_size * beam_size, -1])

            # If next_word_ids is EOS, beam_mask_ should be 0.0
            beam_mask_ = 1.0 - next_word_ids.eq(Vocab.EOS).float()
            # If last step a EOS is already generated, we replace the last token as PAD
            next_word_ids.masked_fill_((beam_mask_ + beam_mask).eq(0.0), Vocab.PAD)
            if copy_attn:
                next_word_ids_copy.masked_fill_((beam_mask_ + beam_mask).eq(0.0), Vocab.PAD)
            beam_mask = beam_mask * beam_mask_

            # update beam
            dec_memory_len += beam_mask

            final_word_indices = torch.cat((final_word_indices, torch.unsqueeze(next_word_ids, 2)), dim=2)

            if copy_attn:
                mask_ext = next_word_ids_copy.ge(vocab_tgt_size).long()
                next_word_ids_copy = next_word_ids_copy * (1.0 - mask_ext) + mask_ext * Vocab.UNK
                final_word_indices_copy = \
                    torch.cat((final_word_indices_copy, torch.unsqueeze(next_word_ids_copy, 2)), dim=2)

            if beam_mask.eq(0.0).all():
                # All the beam is finished (be zero
                break

        scores = beam_scores / (dec_memory_len + 1e-2)

        _, reranked_ids = torch.sort(scores, dim=-1, descending=False)

        return tensor_gather_helper(gather_indices=reranked_ids,
                                    gather_from=final_word_indices[:, :, 1:].contiguous(),
                                    batch_size=batch_size,
                                    beam_size=beam_size,
                                    gather_shape=[batch_size * beam_size, -1])

    def forward(self, src_seq, tgt_seq=None, mode="train", **kwargs):

        if mode == "train":
            assert tgt_seq is not None

            tgt_seq = tgt_seq.transpose(1, 0).contiguous()  # length first

            return self.force_teaching(src_seq, tgt_seq)

        elif mode == "infer":
            with torch.no_grad():
                return self.batch_beam_search(x=src_seq, **kwargs)