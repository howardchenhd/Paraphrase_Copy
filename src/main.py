import time
import os
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm
import codecs

import torch
import numpy as np
from src.utils.common_utils import *
from src.utils.logging import *
from src.utils.data_io import ZipDatasets, TextDataset, DataIterator
from src.metric.bleu_scorer import ExternalScriptBLEUScorer
import src.models
from src.models import *
from src.modules.criterions import NMTCriterion, NMTCopyCriterion
from src.utils.optim import Optimizer
from src.utils.lr_scheduler import LossScheduler, NoamScheduler
from .tokenizeChinese import tokenizePlainFile

# Fix random seed
torch.manual_seed(GlobalNames.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GlobalNames.SEED)


def prepare_data(seqs_x, seqs_y=None, cuda=False, batch_first=True):
    """
    Args:
        eval ('bool'): indicator for eval/infer.

    Returns: [batch, maxlen]

    """

    def _np_pad_batch_2D(samples, pad, batch_first=True, cuda=True):

        batch_size = len(samples)

        sizes = [len(s) for s in samples]
        max_size = max(sizes)

        x_np = np.full((batch_size, max_size), fill_value=pad, dtype='int64')

        for ii in range(batch_size):
            x_np[ii, :sizes[ii]] = samples[ii]

        if batch_first is False:
            x_np = np.transpose(x_np, [1, 0])

        x = torch.tensor(x_np)

        if cuda is True:
            x = x.cuda()
        return x

    seqs_x = list(map(lambda s: [Vocab.BOS] + s[0] + [Vocab.EOS], seqs_x))
    x = _np_pad_batch_2D(samples=seqs_x, pad=Vocab.PAD,
                         cuda=cuda, batch_first=batch_first)

    if seqs_y is None:
        return x

    seqs_y = list(map(lambda s: [Vocab.BOS] + s[0] + [Vocab.EOS], seqs_y))
    y = _np_pad_batch_2D(seqs_y, pad=Vocab.PAD,
                         cuda=cuda, batch_first=batch_first)

    return x, y


def prepare_data_forCopy(seqs_x, vocab_ext, seqs_y=None, cuda=False, batch_first=True, copy_sup=False):
    """
    Args:
        eval ('bool'): indicator for eval/infer.
        vocab_ext: vocab_tgt + (vocab produced by current source batch for copy)
    Returns: [batch, maxlen]

    """

    def _np_pad_batch_2D(samples, pad, batch_first=True, cuda=True):

        batch_size = len(samples)

        sizes = [len(s) for s in samples]
        max_size = max(sizes)

        x_np = np.full((batch_size, max_size), fill_value=pad, dtype='int64')

        for ii in range(batch_size):
            x_np[ii, :sizes[ii]] = samples[ii]

        if batch_first is False:
            x_np = np.transpose(x_np, [1, 0])

        x = torch.tensor(x_np)

        if cuda is True:
            x = x.cuda()
        return x

    # build extended vocab by source words and return source words' indices in the extend vocab
    def extend_vocab(samples):
        sams = []
        for s in samples:
            line_ori = s[1]
            line = []
            for w in line_ori:
                idx = vocab_ext.token2id(w)
                if idx == Vocab.UNK:
                    vocab_ext.update_vocab(w)
                    idx = vocab_ext.max_n_words - 1
                line.append(idx)
            line = [Vocab.BOS] + line + [Vocab.EOS]
            sams.append(line)

        return sams

    def token2id(samples):
        sams = []
        for s in samples:
            line = s[1]
            line = [vocab_ext.token2id(w) for w in line]
            line = [Vocab.BOS] + line + [Vocab.EOS]
            sams.append(line)
        return sams

    def copy_label(s_y, s_x):
        labels = []
        for y_, x_ in zip(s_y, s_x):
            to_copy = set(x_[1])
            label = [1 if _ in to_copy else 0 for _ in y_[1]]
            label = [1] + label + [1]
            labels.append(label)
        return labels

    seqs_x_ext = extend_vocab(seqs_x)
    seqs_x_ext = _np_pad_batch_2D(samples=seqs_x_ext, pad=Vocab.PAD,
                                  cuda=cuda, batch_first=batch_first)

    seqs_x_t = list(map(lambda s: [Vocab.BOS] + s[0] + [Vocab.EOS], seqs_x))
    x = _np_pad_batch_2D(samples=seqs_x_t, pad=Vocab.PAD,
                         cuda=cuda, batch_first=batch_first)

    if seqs_y is None:
        return x, seqs_x_ext

    # used as y_label
    copy_labels = None
    if copy_sup:
        c_labels = copy_label(seqs_y, seqs_x)
        copy_labels = _np_pad_batch_2D(samples=c_labels, pad=Vocab.UNK,
                                       cuda=cuda, batch_first=batch_first)

    seqs_y_ext = token2id(seqs_y)
    seqs_y_ext = _np_pad_batch_2D(samples=seqs_y_ext, pad=Vocab.PAD,
                                  cuda=cuda, batch_first=batch_first)

    seqs_y = list(map(lambda s: [Vocab.BOS] + s[0] + [Vocab.EOS], seqs_y))
    y = _np_pad_batch_2D(seqs_y, pad=Vocab.PAD,
                         cuda=cuda, batch_first=batch_first)

    return x, y, seqs_x_ext, seqs_y_ext, copy_labels


def compute_forward(model,
                    critic,
                    seqs_x,
                    seqs_y,
                    eval=False,
                    normalization=1.0,
                    batch_dim=0,
                    shard_size=-1,
                    n_correctness=False,
                    seqs_x_ext=None,
                    seqs_y_ext=None,
                    n_words_ext=None,
                    c_labels=None,
                    ):
    """
    :type model: nn.Module

    :type critic: NMTCriterion

    """

    if eval:
        model.eval()
        critic.eval()
    else:
        model.train()
        critic.train()

    y_inp = seqs_y[:, :-1].contiguous()
    if seqs_y_ext is not None:
        y_label = seqs_y_ext[:, 1:].contiguous()
    else:
        y_label = seqs_y[:, 1:].contiguous()

    if c_labels is not None:
        copy_labels = c_labels[:, 1:].contiguous()
    else:
        copy_labels = c_labels

    with torch.set_grad_enabled(not eval):

        dec_outs, attn_w, logits_copy = model(seqs_x, y_inp)

        if seqs_x_ext is not None:
            loss = critic(generator=model.generator,
                          shard_size=shard_size,
                          normalization=normalization,
                          batch_dim=batch_dim,
                          eval=eval,
                          dec_outs=dec_outs,
                          labels=y_label,
                          attn_w=attn_w,
                          logits_copy=logits_copy,
                          n_words_ext=n_words_ext,
                          seqs_x_ext=seqs_x_ext,
                          copy_labels=copy_labels)
        else:
            loss = critic(generator=model.generator,
                          shard_size=shard_size,
                          normalization=normalization,
                          batch_dim=batch_dim,
                          eval=eval,
                          dec_outs=dec_outs,
                          labels=y_label,
                          )

    if n_correctness:

        with torch.no_grad():
            
            mask = y_label.ne(Vocab.PAD)

            if seqs_x_ext is not None:
                scores, copy_scores = model.generator(dec_outs.view(-1, dec_outs.size(2)),
                                                      logits_copy.view(-1, logits_copy.size(2)),
                                                      attn_w.view(-1, attn_w.size(2)))

                if n_words_ext > scores.size(-1):
                    expand = scores.new(scores.size(0), n_words_ext - scores.size(-1)).zero_()
                    scores = torch.cat((scores, expand), 1)

                seqs_x_ext = seqs_x_ext.unsqueeze(1)  # [batch_size, 1, len_src]
                seqs_x_ext = seqs_x_ext.repeat(1, dec_outs.size(1), 1)  # [batch_size, bm, len_src]
                seqs_x_ext = seqs_x_ext.view(-1, seqs_x_ext.size(2))  # [batch_size * seq_len, len_src]
                scores.scatter_add_(1, seqs_x_ext, copy_scores)

                scores = torch.log(scores)
                pred = scores.view(attn_w.size(1), attn_w.size(0), -1).max(2)[1]
            else:
                pred = model.generator(dec_outs).max(2)[1]  # [batch_size, seq_len]

            num_correct = y_label.eq(pred).float().masked_select(mask).sum() / normalization

        return loss.item(), num_correct

    return loss.item()


def loss_validation(model, critic, valid_iterator, copy_attn=False, vocab_ext=None, copy_sup=False):
    """
    :type model: Transformer

    :type critic: NMTCriterion

    :type valid_iterator: DataIterator

    :type copy_attn: whether use copy_attn or not

    :type vocab_ext: extended vocab
    """

    if copy_attn:
        assert vocab_ext is not None, 'if use copy_attn, then extend vocab is needed'

    if copy_sup:
        assert copy_attn, 'if use copy_sup, then copy_attn is needed'

    n_sents = 0
    n_tokens = 0.0

    sum_loss = 0.0
    sum_correct = 0.0

    valid_iter = valid_iterator.build_generator()

    for batch in valid_iter:
        seqs_x, seqs_y = batch

        n_sents += len(seqs_x)
        n_tokens += sum(len(s) for s in seqs_y)

        x_ext = None
        y_ext = None
        n_words_ext = None
        copy_labels = None
        if copy_attn:
            x, y, x_ext, y_ext, copy_labels = prepare_data_forCopy(seqs_x, vocab_ext, seqs_y, cuda=GlobalNames.USE_GPU,
                                                                   copy_sup=copy_sup)
            n_words_ext = vocab_ext.max_n_words
        else:
            x, y = prepare_data(seqs_x, seqs_y, cuda=GlobalNames.USE_GPU)
        # x, y = prepare_data(seqs_x, seqs_y, cuda=GlobalNames.USE_GPU)

        loss, num_correct = compute_forward(model=model,
                                            critic=critic,
                                            seqs_x=x,
                                            seqs_y=y,
                                            eval=True, n_correctness=True,
                                            seqs_x_ext=x_ext,
                                            seqs_y_ext=y_ext,
                                            n_words_ext=n_words_ext,
                                            c_labels=copy_labels
                                            )

        if np.any(np.isnan(loss)):
            WARN("NaN detected!")

        sum_loss += float(loss)
        sum_correct += num_correct

        if copy_attn:
            vocab_ext.reset()

    return float(sum_loss / n_sents), float(sum_correct * 1.0 / n_tokens)


def bleu_validation(uidx,
                    valid_iterator,
                    model,
                    bleu_scorer,
                    vocab_tgt,
                    batch_size,
                    eval_at_char_level=False,
                    valid_dir="./valid",
                    max_steps=10,
                    copy_attn=False,
                    vocab_ext=None
                    ):
    """
    :type model: Transformer

    :type valid_iterator: DataIterator

    :type bleu_scorer: ExternalScriptBLEUScorer

    :type vocab_tgt: Vocab
    """

    if copy_attn:
        assert vocab_ext is not None, 'if use copy_attn, then extend vocab is needed'

    def _split_into_chars(line):
        new_line = []
        for w in line:
            if vocab_tgt.token2id(w) not in {Vocab.UNK, Vocab.EOS, Vocab.BOS, Vocab.PAD}:
                # if not UNK, split into characters
                new_line += list(w)
            else:
                # if UNK, treat as a special character
                new_line.append(w)

        return new_line

    model.eval()

    trans = []

    infer_progress_bar = tqdm(total=len(valid_iterator),
                              desc=' - (Infer)  ',
                              unit="sents")

    valid_iter = valid_iterator.build_generator(batch_size=batch_size)

    for batch in valid_iter:

        seqs_x = batch[0]
        infer_progress_bar.update(len(seqs_x))

        x_ext = None
        n_words_ext = None
        if copy_attn:
            x, x_ext = prepare_data_forCopy(seqs_x, vocab_ext, cuda=GlobalNames.USE_GPU)
            n_words_ext = vocab_ext.max_n_words
        else:
            x = prepare_data(seqs_x, cuda=GlobalNames.USE_GPU)

        word_ids = model(x, mode="infer", beam_size=5, max_steps=max_steps, copy_attn=copy_attn,
                         x_ext=x_ext, n_words_ext=n_words_ext)

        word_ids = word_ids.cpu().numpy().tolist()

        # Append result
        for sent_t in word_ids:
            sent_t = [[wid for wid in line if wid != Vocab.PAD] for line in sent_t]
            x_tokens = []

            for wid in sent_t[0]:
                if wid == Vocab.EOS:
                    break
                if copy_attn:
                    x_tokens.append(vocab_ext.id2token(wid))
                else:
                    x_tokens.append(vocab_tgt.id2token(wid))

            if len(x_tokens) > 0:
                trans.append(' '.join(x_tokens))
            else:
                if copy_attn:
                    trans.append('%s' % vocab_ext.id2token(Vocab.EOS))
                else:
                    trans.append('%s' % vocab_tgt.id2token(Vocab.EOS))
        if copy_attn:
            vocab_ext.reset()

    infer_progress_bar.close()

    # Merge bpe segmentation
    trans = [line.replace("@@ ", "") for line in trans]

    # Split into characters
    if eval_at_char_level is True:
        trans = [' '.join(_split_into_chars(line.strip().split())) for line in trans]

    if not os.path.exists(valid_dir):
        os.mkdir(valid_dir)
    
    hyp_path = os.path.join(valid_dir, 'trans.iter{0}.txt'.format(uidx))

    # with open(hyp_path, 'w') as f:
    with codecs.open(hyp_path, 'w', 'utf-8') as f:
        for line in trans:
            f.write('%s\n' % line)

    # lang = bleu_scorer.getlang()
    # postprocess = bleu_scorer.getpostprocess()

    # if lang == 'zh' and postprocess:
    #    tokenizePlainFile(codecs.open(hyp_path, 'r', 'utf-8'), codecs.open(hyp_path + '.char', 'w', 'utf-8'))

    # with open(hyp_path) as f:
    with codecs.open(hyp_path, 'r', 'utf-8') as f:
        bleu_v = bleu_scorer.corpus_bleu(f)

    return bleu_v


def load_pretrained_model(nmt_model, pretrain_path, map_dict=None, exclude_prefix=None, device: str = "cpu"):
    """
    Args:
        nmt_model: model.
        pretrain_path ('str'): path to pretrained model.
        map_dict ('dict'): mapping specific parameter names to those names
            in current model.
        exclude_prefix ('dict'): excluding parameters with specific names
            for pretraining.

    Raises:
        ValueError: Size not match, parameter name not match or others.

    """
    if exclude_prefix is None:
        exclude_prefix = []
    if pretrain_path != "":
        INFO("Loading pretrained model from {}".format(pretrain_path))
        pretrain_params = torch.load(pretrain_path, map_location=device)
        for name, params in pretrain_params.items():
            flag = False
            for pp in exclude_prefix:
                if name.startswith(pp):
                    flag = True
                    break
            if flag:
                continue
            INFO("Loading param: {}...".format(name))
            try:
                nmt_model.load_state_dict({name: params}, strict=False)
            except Exception as e:
                WARN("{}: {}".format(str(Exception), e))

        INFO("Pretrained model loaded.")


def train(FLAGS):
    """
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    write_log_to_file(os.path.join(FLAGS.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))

    GlobalNames.USE_GPU = FLAGS.use_gpu

    if GlobalNames.USE_GPU:
        GlobalNames.CURRENT_DEVICE = "cuda:0"

    config_path = os.path.abspath(FLAGS.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    if "seed" in training_configs:
        # Set random seed
        GlobalNames.SEED = training_configs['seed']

    if 'buffer_size' not in training_configs:
        training_configs['buffer_size'] = 100 * training_configs['batch_size']

    saveto_collections = '%s.pkl' % os.path.join(FLAGS.saveto, FLAGS.model_name + GlobalNames.MY_CHECKPOINIS_PREFIX)
    saveto_best_model = os.path.join(FLAGS.saveto, FLAGS.model_name + GlobalNames.MY_BEST_MODEL_SUFFIX)
    saveto_best_optim_params = os.path.join(FLAGS.saveto,
                                            FLAGS.model_name + GlobalNames.MY_BEST_OPTIMIZER_PARAMS_SUFFIX)

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocab(dict_path=data_configs['dictionaries'][0], max_n_words=data_configs['n_words'][0])
    vocab_tgt = Vocab(dict_path=data_configs['dictionaries'][1], max_n_words=data_configs['n_words'][1])

    vocab_ext = None
    if model_configs['copy_attn']:
        vocab_ext = VocabCopy(vocab=vocab_tgt)

    train_bitext_dataset = ZipDatasets(
        TextDataset(data_path=data_configs['train_data'][0],
                    vocab=vocab_src,
                    bpe_codes=data_configs['bpe_codes'][0],
                    max_len=data_configs['max_len'][0],
                    use_char=data_configs['use_char'][0]
                    ),
        TextDataset(data_path=data_configs['train_data'][1],
                    vocab=vocab_tgt,
                    bpe_codes=data_configs['bpe_codes'][1],
                    max_len=data_configs['max_len'][1],
                    use_char=data_configs['use_char'][1]
                    ),
        shuffle=training_configs['shuffle']
    )

    valid_bitext_dataset = ZipDatasets(
        TextDataset(data_path=data_configs['valid_data'][0],
                    vocab=vocab_src,
                    bpe_codes=data_configs['bpe_codes'][0],
                    use_char=data_configs['use_char'][0]
                    ),
        TextDataset(data_path=data_configs['valid_data'][1],
                    vocab=vocab_tgt,
                    bpe_codes=data_configs['bpe_codes'][1],
                    use_char=data_configs['use_char'][1]
                    )
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=training_configs['batch_size'],
                                     sort_buffer=training_configs['use_bucket'],
                                     buffer_size=training_configs['buffer_size'],
                                     sort_fn=lambda line: len(line[-1]))

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  sort_buffer=False)

    bleu_scorer = ExternalScriptBLEUScorer(reference_path=data_configs['bleu_valid_reference'],
                                           lang=data_configs['lang_pair'].split('-')[1],
                                           bleu_script=training_configs['bleu_valid_configs']['bleu_script'],
                                           digits_only=True,
                                           lc=training_configs['bleu_valid_configs']['lowercase'],
                                           postprocess=training_configs['bleu_valid_configs']['postprocess']
                                           )

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    model_collections = Collections()

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================================================================== #
    # Build Model & Sampler & Validation
    INFO('Building model...')
    timer.tic()

    model_cls = model_configs.get("model")
    if model_cls not in src.models.__all__:
        raise ValueError(
            "Invalid model class \'{}\' provided. Only {} are supported now.".format(
                model_cls, src.models.__all__))

    nmt_model = eval(model_cls)(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **model_configs)
    INFO(nmt_model)

    if model_configs['copy_attn']:
        critic = NMTCopyCriterion(label_smoothing=model_configs['label_smoothing'], copy_sup=model_configs['copy_sup'])
    else:
        critic = NMTCriterion(label_smoothing=model_configs['label_smoothing'])

    INFO(critic)
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    if GlobalNames.USE_GPU:
        nmt_model = nmt_model.cuda()
        critic = critic.cuda()

    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=nmt_model,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    # Configure Learning Scheduler
    # Here we have two policies, "loss" and "noam"

    if optimizer_configs['schedule_method'] is not None:

        if optimizer_configs['schedule_method'] == "loss":

            scheduler = LossScheduler(optimizer=optim, **optimizer_configs['scheduler_configs']
                                  )

        elif optimizer_configs['schedule_method'] == "noam":
            scheduler = NoamScheduler(optimizer=optim, **optimizer_configs['scheduler_configs'])
        else:
            WARN("Unknown scheduler name {0}. Do not use lr_scheduling.".format(optimizer_configs['schedule_method']))
            scheduler = None
    else:
        scheduler = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # Initialize training indicators
    uidx = 0
    bad_count = 0

    # Whether Reloading model
    if FLAGS.reload is True and os.path.exists(saveto_best_model):
        timer.tic()
        INFO("Reloading model...")
        params = torch.load(saveto_best_model, map_location=GlobalNames.CURRENT_DEVICE)
        nmt_model.load_state_dict(params)

        model_archives = Collections.unpickle(path=saveto_collections)
        model_collections.load(archives=model_archives)

        uidx = model_archives['uidx']
        bad_count = model_archives['bad_count']

        INFO("Done. Model reloaded.")

        if os.path.exists(saveto_best_optim_params):
            INFO("Reloading optimizer params...")
            optimizer_params = torch.load(saveto_best_optim_params, map_location=GlobalNames.CURRENT_DEVICE)
            optim.optim.load_state_dict(optimizer_params)

            INFO("Done. Optimizer params reloaded.")
        elif uidx > 0:
            INFO("Failed to reload optimizer params: {} does not exist".format(
                saveto_best_optim_params))

        INFO('Done. Elapsed time {0}'.format(timer.toc()))
    # New training. Check if pretraining needed
    else:
        # pretrain
        load_pretrained_model(nmt_model, FLAGS.pretrain_path, exclude_prefix=None, device=GlobalNames.CURRENT_DEVICE)

    # ================================================================================== #
    # Prepare training

    params_best_loss = None

    summary_writer = SummaryWriter(log_dir=FLAGS.log_path)

    cum_samples = 0
    cum_words = 0
    valid_loss = 1.0 * 1e12  # Max Float
    saving_files = []

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    # lang = bleu_scorer.getlang()
    # postprocess = bleu_scorer.getpostprocess()
    #
    # hyp_path = '/Users/chenhuadong/code/data/devtest/cwmt2008/cwmt2008.src'
    # if lang == 'zh' and postprocess:
    #    tokenizePlainFile(codecs.open(hyp_path, 'r', 'utf-8'), codecs.open(hyp_path + '.char', 'w', 'utf-8'))
    #
    # with open(hyp_path + '.char') as f:
    #       bleu_v = bleu_scorer.corpus_bleu(f)

    for eidx in range(training_configs['max_epochs']):
        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)

        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents"
                                     )
        for batch in training_iter:

            uidx += 1

            # ================================================================================== #
            # Learning rate annealing

            if scheduler is not None and (np.mod(uidx, scheduler.schedule_freq) == 0 or FLAGS.debug):

                if scheduler.step(global_step=uidx, loss=valid_loss):

                    if optimizer_configs['schedule_method'] == "loss":
                        nmt_model.load_state_dict(params_best_loss)

                new_lr = list(optim.get_lrate())[0]
                summary_writer.add_scalar("lrate", new_lr, global_step=uidx)

            seqs_x, seqs_y = batch

            batch_size_t = len(seqs_x)
            cum_samples += batch_size_t
            cum_words += sum(len(s) for s in seqs_y)

            training_progress_bar.update(batch_size_t)

            # Prepare data
            # x, y = prepare_data(seqs_x, seqs_y, cuda=GlobalNames.USE_GPU)
            x_ext = None
            y_ext = None
            n_words_ext = None
            copy_labels = None
            if model_configs['copy_attn']:
                x, y, x_ext, y_ext, copy_labels = prepare_data_forCopy(seqs_x, vocab_ext, seqs_y,
                                                                       cuda=GlobalNames.USE_GPU,
                                                                       copy_sup=model_configs['copy_sup'])
                n_words_ext = vocab_ext.max_n_words
            else:
                x, y = prepare_data(seqs_x, seqs_y, cuda=GlobalNames.USE_GPU)

            # optim.zero_grad()
            nmt_model.zero_grad()
            loss = compute_forward(model=nmt_model,
                                   critic=critic,
                                   seqs_x=x,
                                   seqs_y=y,
                                   eval=False,
                                   normalization=batch_size_t,
                                   shard_size=training_configs['shard_size'],
                                   seqs_x_ext=x_ext,
                                   seqs_y_ext=y_ext,
                                   n_words_ext=n_words_ext,
                                   c_labels=copy_labels)
            optim.step()

            # ================================================================================== #
            # Display some information
            if np.mod(uidx, training_configs['disp_freq']) == 0:
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0

            # ================================================================================== #
            # Saving checkpoints
            if np.mod(uidx, training_configs['save_freq']) == 0 or FLAGS.debug:

                if not os.path.exists(FLAGS.saveto):
                    os.mkdir(FLAGS.saveto)

                INFO('Saving the model at iteration {}...'.format(uidx))

                if not os.path.exists(FLAGS.saveto):
                    os.mkdir(FLAGS.saveto)

                saveto_uidx = os.path.join(FLAGS.saveto, FLAGS.model_name + '.iter%d.tpz' % uidx)
                torch.save(nmt_model.state_dict(), saveto_uidx)

                Collections.pickle(path=saveto_collections,
                                   uidx=uidx,
                                   bad_count=bad_count,
                                   **model_collections.export())

                saving_files.append(saveto_uidx)

                INFO('Done')

                if len(saving_files) > 5:
                    for f in saving_files[:-1]:
                        os.remove(f)

                    saving_files = [saving_files[-1]]


            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if np.mod(uidx, training_configs['loss_valid_freq']) == 0 or FLAGS.debug:

                if model_configs['copy_attn']:
                    vocab_ext.reset()
                valid_loss, valid_n_correct = loss_validation(model=nmt_model,
                                                              critic=critic,
                                                              valid_iterator=valid_iterator,
                                                              copy_attn=model_configs['copy_attn'],
                                                              vocab_ext=vocab_ext,
                                                              copy_sup=model_configs['copy_sup'],
                                                              )

                model_collections.add_to_collection("history_losses", valid_loss)

                min_history_loss = np.array(model_collections.get_collection("history_losses")).min()

                summary_writer.add_scalar("loss", valid_loss, global_step=uidx)
                summary_writer.add_scalar("best_loss", min_history_loss, global_step=uidx)
                summary_writer.add_scalar("n_correct", valid_n_correct, global_step=uidx)

                # If no bess loss model saved, save it.
                if len(model_collections.get_collection("history_losses")) == 0 or params_best_loss is None:
                    params_best_loss = nmt_model.state_dict()

                if valid_loss <= min_history_loss:
                    params_best_loss = nmt_model.state_dict()  # Export best variables

            # ================================================================================== #
            # BLEU Validation & Early Stop

            if (np.mod(uidx, training_configs['bleu_valid_freq']) == 0 and uidx > training_configs['bleu_valid_warmup']) \
                    or FLAGS.debug:
                if model_configs['copy_attn']:
                    vocab_ext.reset()
                valid_bleu = bleu_validation(uidx=uidx,
                                             valid_iterator=valid_iterator,
                                             batch_size=training_configs['bleu_valid_batch_size'],
                                             model=nmt_model,
                                             bleu_scorer=bleu_scorer,
                                             eval_at_char_level=data_configs['eval_at_char_level'],
                                             vocab_tgt=vocab_tgt,
                                             valid_dir=FLAGS.valid_path,
                                             max_steps=training_configs["bleu_valid_max_steps"],
                                             copy_attn=model_configs['copy_attn'],
                                             vocab_ext=vocab_ext
                                             )

                model_collections.add_to_collection(key="history_bleus", value=valid_bleu)

                best_valid_bleu = float(np.array(model_collections.get_collection("history_bleus")).max())

                summary_writer.add_scalar("bleu", valid_bleu, uidx)
                summary_writer.add_scalar("best_bleu", best_valid_bleu, uidx)

                # If model get new best valid bleu score
                if valid_bleu >= best_valid_bleu:
                    bad_count = 0

                    if is_early_stop is False:
                        INFO('Saving best model...')

                        # save model
                        best_params = nmt_model.state_dict()
                        torch.save(best_params, saveto_best_model)

                        # save optim params
                        INFO('Saving best optimizer params...')
                        best_optim_params = optim.optim.state_dict()
                        torch.save(best_optim_params, saveto_best_optim_params)

                        INFO('Done.')

                else:
                    bad_count += 1

                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                with open("./valid.txt", 'a') as f:
                    f.write("{0} Loss: {1:.2f} BLEU: {2:.2f} lrate: {3:6f} patience: {4}\n".format(uidx, valid_loss,
                                                                                                   valid_bleu, lrate,
                                                                                                   bad_count))
            if model_configs['copy_attn']:
                vocab_ext.reset()

        training_progress_bar.close()


def translate(FLAGS):
    GlobalNames.USE_GPU = FLAGS.use_gpu

    config_path = os.path.abspath(FLAGS.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocab(dict_path=data_configs['dictionaries'][0], max_n_words=data_configs['n_words'][0])
    vocab_tgt = Vocab(dict_path=data_configs['dictionaries'][1], max_n_words=data_configs['n_words'][1])

    if model_configs['copy_attn']:
        vocab_ext = VocabCopy(vocab=vocab_tgt)

    valid_dataset = TextDataset(data_path=FLAGS.source_path,
                                vocab=vocab_src,
                                bpe_codes=data_configs['bpe_codes'][0])

    valid_iterator = DataIterator(dataset=valid_dataset,
                                  batch_size=FLAGS.batch_size,
                                  sort_buffer=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #
    # Build Model & Sampler & Validation
    INFO('Building model...')
    timer.tic()

    model_cls = model_configs.get("model")
    if model_cls not in src.models.__all__:
        raise ValueError(
            "Invalid model class \'{}\' provided. Only {} are supported now.".format(
                model_cls, src.models.__all__))

    nmt_model = eval(model_cls)(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **model_configs)

    nmt_model.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()

    params = torch.load(FLAGS.model_path, map_location="cpu")

    nmt_model.load_state_dict(params)

    if GlobalNames.USE_GPU:
        nmt_model.cuda()

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Begin...')

    result = []
    n_words = 0

    timer.tic()

    infer_progress_bar = tqdm(total=len(valid_iterator),
                              desc=' - (Infer)  ',
                              unit="sents")

    valid_iter = valid_iterator.build_generator()
    for batch in valid_iter:

        seqs_x = batch[0]

        batch_size_t = len(seqs_x)

        if model_configs['copy_attn']:
            x, x_ext = prepare_data_forCopy(seqs_x, vocab_ext, cuda=GlobalNames.USE_GPU)
            word_ids = nmt_model(x, mode="infer", beam_size=5, max_steps=FLAGS.max_steps,
                                 copy_attn=True, x_ext=x_ext, n_words_ext=vocab_ext.max_n_words)
        else:
            x = prepare_data(seqs_x=seqs_x, cuda=GlobalNames.USE_GPU)
            word_ids = nmt_model(x, mode="infer", beam_size=5, max_steps=FLAGS.max_steps)

        word_ids = word_ids.cpu().numpy().tolist()

        # Append result
        for sent_t in word_ids:
            sent_t = [[wid for wid in line if wid != Vocab.PAD] for line in sent_t]
            result.append(sent_t)

            n_words += len(sent_t[0])

        infer_progress_bar.update(batch_size_t)

    infer_progress_bar.close()

    INFO('Done. Speed: {0:.2f} words/sec'.format(n_words / (timer.toc(return_seconds=True))))

    translation = []
    for sent in result:
        samples = []
        for trans in sent:
            sample = []
            for w in trans:
                if w == Vocab.EOS:
                    break
                if model_configs['copy_attn']:
                    sample.append(vocab_ext.id2token(w))
                else:
                    sample.append(vocab_tgt.id2token(w))
            samples.append(' '.join(sample))
        translation.append(samples)

    keep_n = FLAGS.beam_size if FLAGS.keep_n <= 0 else min(FLAGS.beam_size, FLAGS.keep_n)
    outputs = ['%s.%d' % (FLAGS.saveto, i) for i in range(keep_n)]

    with batch_open(outputs, 'w') as handles:
        for trans in translation:
            for i in range(keep_n):
                if i < len(trans):
                    handles[i].write('%s\n' % trans[i])
                else:
                    handles[i].write('%s\n' % 'eos')
