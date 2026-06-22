"""
Self-contained reproduction of the TEXT decoding result from
"A high-performance neuroprosthesis for speech decoding and avatar control".

This is a faithful, script-ified version of text/text_example_decoding.ipynb:
  1. Loads the 1024-word-General training data + the 249 realtime test sentences.
  2. Normalizes, builds the phone lexicon/tokens, sets up CTC targets.
  3. Trains a CNN+BiGRU CTC model (the AUXCnnRnnClassifier from the paper).
  4. Evaluates on the *unseen* realtime test sentences with early stopping and a
     CTC beam search + n-gram language model, then computes WER / CER / PER /
     words-per-minute and the pseudo-blocked median WER (the headline metric).

It is intentionally free of `wandb` and `speechbrain` dependencies: the model is
defined inline and only the dependency-free helpers from the repo are imported.

All progress is printed to stdout; run it under `submit_job -o logs/<name>.txt`
so the full training log is captured in the logs/ folder.

Example (defaults reproduce the notebook's run):
    python repro/train_text_decoder.py \
        --data_dir /userdata/dmoses/b3_features/zenodo \
        --out_dir  repro/runs/run1 \
        --lm_3gram text/custom_lms/full_corpus_lm_3_abs_slm.binary \
        --lm_5gram text/custom_lms/full_corpus_lm_5_abs_slm.binary
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from torchvision import transforms

# ---------------------------------------------------------------------------
# Make the repo's text/ folder importable (this file lives in repo_root/repro/)
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_DIR = os.path.join(REPO_ROOT, "text")
sys.path.insert(0, TEXT_DIR)

import torchaudio
from torchaudio.models.decoder import ctc_decoder

# dependency-free helpers from the repo
from train.ctc_decoding import GreedyCTCDecoder, greedy_beam_wer_cer
from data_loading_utilities.normalize import (
    normalize, minmax_scaling, pertrial_minmax, rezscore,
)
from data_loading_utilities.clean_labels import clean_labels
from data_loading_utilities.torch_ctc_loaders import (
    CTCDataset_Wordct, Jitter, Blackout, AdditiveNoise, LevelChannelNoise,
    ScaleAugment,
)


# ===========================================================================
# Model (copied from text/models/cnn_rnn_w_aux.py so we don't import
# speechbrain, which that module pulls in at import time).
# ===========================================================================
class AUXCnnRnnClassifier(nn.Module):
    """CNN front-end + bidirectional GRU that emits a phone distribution per
    time step, plus an auxiliary word-count head (unused, weight 0 by default)."""

    def __init__(self, rnn_dim, KS, num_layers, dropout, n_targ, bidirectional,
                 in_channels=506, nword_targ=10):
        super().__init__()
        self.preprocessing_conv = nn.Conv1d(in_channels=in_channels,
                                             out_channels=rnn_dim,
                                             kernel_size=KS, stride=KS)
        self.BiGRU = nn.GRU(input_size=rnn_dim, hidden_size=rnn_dim,
                            num_layers=num_layers, bidirectional=bidirectional,
                            dropout=dropout)
        self.num_layers = num_layers
        self.rnn_dim = rnn_dim
        self.ks = KS
        self.dropout = nn.Dropout(dropout)
        mult = 2 if bidirectional else 1
        self.mult = mult
        self.dense = nn.Linear(rnn_dim * mult, n_targ)
        self.word_ct_layer = nn.Linear(rnn_dim * mult, nword_targ)

    def forward(self, x, lens):
        lens = torch.div(lens, self.ks, rounding_mode="trunc")
        x = x.contiguous().permute(0, 2, 1)        # B,C,T for conv
        x = self.preprocessing_conv(x)
        x = self.dropout(x)
        x = x.contiguous().permute(2, 0, 1)        # T,B,C for rnn
        packed = pack_padded_sequence(x, lens.int().cpu(), enforce_sorted=False)
        emissions, _ = self.BiGRU(packed)
        unpacked_emissions, lens_unpacked = pad_packed_sequence(emissions)
        unpacked_for_wordct = unpacked_emissions[-1]
        unpacked_outputs = self.dense(unpacked_emissions)
        return unpacked_outputs, self.word_ct_layer(unpacked_for_wordct), lens_unpacked


# ===========================================================================
# Training / evaluation helpers (no wandb)
# ===========================================================================
def train_one_epoch(model, loader, optimizer, device, wordct_weight, clipamt):
    model.train()
    loss_fn = nn.CTCLoss()
    class_loss = nn.CrossEntropyLoss()
    total_loss, total_samps = 0.0, 0
    for x, y, l, targ_len, _, wordct in loader:
        x = x.float().to(device)
        y = y.long().to(device)
        l = l.int().cpu()
        targ_len = targ_len.int().cpu()
        wordct = wordct.long().to(device)
        emissions, word_ct_pred, lengths = model(x, l)
        emissions = F.log_softmax(emissions, dim=-1)
        optimizer.zero_grad()
        loss = loss_fn(emissions, y, lengths, targ_len)
        if wordct_weight:
            loss = loss + wordct_weight * class_loss(word_ct_pred, wordct)
        total_loss += loss.item()
        total_samps += x.shape[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipamt)
        optimizer.step()
    return total_loss / max(total_samps, 1)


@torch.no_grad()
def evaluate(model, loader, device, greedy, beam_search_decoder, texts, tokens,
             compute_text=False, verbose=False, n_print=10):
    """Returns (loss, wer, cer, per) on the loader. wer/cer/per only when
    compute_text=True (it runs the beam search, which is slow)."""
    model.eval()
    loss_fn = nn.CTCLoss()
    total_loss, total_samps = 0.0, 0
    total_wer = total_cer = total_per = total_gper = 0.0
    gts, transcripts = [], []
    for x, y, l, targ_len, gtsent, _ in loader:
        x = x.float().to(device)
        y = y.long().to(device)
        l = l.int().to(device)
        gtsent = gtsent.long().cpu().numpy()
        targ_len = targ_len.int().to(device)
        emissions, _, lengths = model(x, l)
        emissions = F.log_softmax(emissions, dim=-1)
        total_loss += loss_fn(emissions, y, lengths, targ_len).item()
        total_samps += x.shape[0]
        if compute_text:
            for k, e in enumerate(emissions.permute(1, 0, 2)):
                gt_phones = [tokens[yy] for yy in y.detach().cpu().numpy()[k]
                             if yy != -1 and yy != 0]
                gt = texts[int(gtsent[k])]
                gper, gt_, transcript, cer, wer, per = greedy_beam_wer_cer(
                    e, gt, gt_phones, greedy, beam_search_decoder)
                gts.append(gt_)
                transcripts.append(transcript)
                total_wer += wer
                total_cer += cer
                total_per += per
                total_gper += gper
    loss = total_loss / max(total_samps, 1)
    if not compute_text:
        return loss, None, None, None
    if verbose:
        for gt, tr in list(zip(gts, transcripts))[:n_print]:
            print(f"    gt: {gt}\n    tr: {tr}", flush=True)
    return (loss, total_wer / total_samps, total_cer / total_samps,
            total_per / total_samps)


def train_loop(model, train_loader, val_loader, optimizer, device, texts,
               greedy, beam_search_decoder, tokens, out_dir, max_epochs=1000,
               patience=10, wercalcrate=3, start_eval=0, wordct_weight=0.0,
               clipamt=1.0):
    best_wer = np.inf
    best_state = None
    patience_ctr = 0
    for epoch in range(max_epochs):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, device,
                                  wordct_weight, clipamt)
        do_text = (epoch % wercalcrate == 0 and epoch >= start_eval)
        te_loss, wer, cer, per = evaluate(
            model, val_loader, device, greedy, beam_search_decoder, texts,
            tokens, compute_text=do_text, verbose=do_text)
        msg = (f"epoch {epoch:4d}  tr_loss {tr_loss:.4f}  val_loss {te_loss:.4f}"
               f"  ({time.time() - t0:.1f}s)")
        if wer is not None:
            msg += f"  WER {wer:.4f}  CER {cer:.4f}  PER {per:.4f}  best {min(best_wer,1):.4f}"
        print(msg, flush=True)
        if wer is not None:
            if wer < best_wer:
                best_wer = wer
                patience_ctr = 0
                best_state = copy.deepcopy(model.state_dict())
                if wer < 0.85:
                    torch.save(best_state, os.path.join(out_dir, "best_model.pth"))
                    print(f"    saved best_model.pth (WER {wer:.4f})", flush=True)
            else:
                patience_ctr += 1
            if patience_ctr > patience:
                print(f"early stopping at epoch {epoch} (patience {patience})", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_wer


def check_1024_done(emissions):
    """Early-stop criterion: last 8 steps are confidently silence/blank."""
    chk = emissions[-8:, :2]
    return torch.mean(torch.sum(chk, dim=-1)) > 7.1 / 8


@torch.no_grad()
def eval_realtime_wpm(model, loader, device, greedy, beam_search_decoder, texts,
                      tokens, verbose=True):
    """Realtime-style evaluation with early stopping (batch size must be 1).
    Returns dict with per-sentence wer/cer/per/wpm and aggregate metrics."""
    model.eval()
    model.to(device)
    wers, cers, pers, wpms, speak_times = [], [], [], [], []
    gts, transcripts = [], []
    total_wer = total_cer = total_per = 0.0
    total_samps = 0
    for x, y, l, targ_len, gtsent, _ in loader:
        assert x.shape[0] == 1, "realtime WPM eval needs batch_size=1"
        x = x.float().to(device)
        y = y.long().to(device)
        gtsent = gtsent.long().cpu().numpy()

        t_elapsed = 7.5
        emissions = None
        for t_elapsed in [1.9, 2.7, 3.5, 4.3, 5.1, 5.9, 6.7, 7.5]:
            sample_ct = int(((t_elapsed + 0.5) * 200) / 6)
            l_cur = torch.tensor([sample_ct]).int().to(device)
            emissions, _, _ = model(x[:, :sample_ct], l_cur)
            emissions = F.softmax(emissions, dim=-1)
            if check_1024_done(emissions.squeeze()):
                break
        if t_elapsed < 7.5:
            silent_time = (8 * 4 * 6) / 200  # accounts for the trailing silence window
            speaking_time = t_elapsed - silent_time
        else:
            speaking_time = t_elapsed
        speak_times.append(speaking_time)

        emissions = torch.log(emissions)
        total_samps += 1
        for k, e in enumerate(emissions.permute(1, 0, 2)):
            gt_phones = [tokens[yy] for yy in y.detach().cpu().numpy()[k]
                         if yy != -1 and yy != 0]
            gt = texts[int(gtsent[k])]
            gper, gt_, transcript, cer, wer, per = greedy_beam_wer_cer(
                e, gt, gt_phones, greedy, beam_search_decoder)
            gts.append(gt_)
            transcripts.append(transcript)
            n_words = len(transcript.strip().split(" ")) if transcript.strip() else 0
            wpms.append(60 * n_words / speaking_time if speaking_time > 0 else 0.0)
            wers.append(wer)
            cers.append(cer)
            pers.append(per)
            total_wer += wer
            total_cer += cer
            total_per += per

    if verbose:
        print("net wer", total_wer / total_samps, flush=True)
        print("net cer", total_cer / total_samps, flush=True)
        print("net per", total_per / total_samps, flush=True)
    return {
        "wers": wers, "cers": cers, "pers": pers, "wpms": wpms,
        "speak_times": speak_times, "gts": gts, "transcripts": transcripts,
        "net_wer": total_wer / total_samps, "net_cer": total_cer / total_samps,
        "net_per": total_per / total_samps,
    }


def parceled_metric(vals, lengths=None, parcelation=10):
    """Pseudo-blocking: weighted-average a metric over blocks of `parcelation`
    sentences (weights = sentence word counts), matching the notebook."""
    dist, cur_list, cur_lengths = [], [], []
    if lengths is None:
        lengths = np.ones(len(vals))
    for k, x in enumerate(vals):
        cur_list.append(x)
        cur_lengths.append(np.nan if pd.isna(x) else lengths[k])
        if (k + 1) % parcelation == 0:
            num = np.nansum(np.array(cur_list) * np.array(cur_lengths))
            dist.append(num / np.nansum(np.array(cur_lengths)))
            cur_list, cur_lengths = [], []
    if len(cur_list) > parcelation / 2:
        dist.append(np.nansum(np.array(cur_list) * np.array(cur_lengths))
                    / np.nansum(np.array(cur_lengths)))
    return np.array(dist)


# ===========================================================================
# Data preparation (mirrors the notebook)
# ===========================================================================
def normalize_X(X, strategy):
    if strategy == "typical":
        X[:, :, :X.shape[-1] // 2] = normalize(X[:, :, :X.shape[-1] // 2])
        X[:, :, X.shape[-1] // 2:] = normalize(X[:, :, X.shape[-1] // 2:])
    elif strategy == "norm_all_at_once":
        X = normalize(X)
    elif strategy == "norm_times":
        X = normalize(X, axis=1)
    elif strategy == "zero_to_one":
        X = minmax_scaling(X)
    elif strategy == "rezscore":
        X = rezscore(X)
    elif strategy == "pertrial_minmax":
        X = pertrial_minmax(X)
    elif strategy == "none":
        print("no normalization.")
    return X


def build_lexicon_and_tokens(labels, out_ctc_dir):
    os.makedirs(out_ctc_dir, exist_ok=True)
    labels, all_ph = clean_labels(labels)
    phone_enc = {v: k for k, v in enumerate(
        sorted([a for a in list(set(all_ph)) if a != "|"]))}

    lex = {}
    for k, v in zip(labels["txt_label"], labels["ph_label"]):
        if "|" not in v:
            lex[k] = " ".join(v) + " |"
        else:
            v = "_".join(v).split("|")
            for kk, vv in zip(k.split(" "), v):
                vv = vv.replace("_", " ").strip() + " |"
                if kk != "":
                    lex[kk] = vv

    strings = [k + " " + str(v) for k, v in lex.items()]
    strings = [s for s in strings if len(s) > 3]
    lex_path = os.path.join(out_ctc_dir, "lexicon_phrases_1k.txt")
    with open(lex_path, "w") as f:
        f.writelines([s + "\n" for s in strings])

    tokens = ["-", "|"] + list(phone_enc.keys())
    tok_path = os.path.join(out_ctc_dir, "tokens_phrases_1k.txt")
    with open(tok_path, "w") as f:
        f.writelines([t + "\n" for t in tokens])

    enc_final = {v: k for k, v in enumerate(tokens)}
    print(f"vocab size: {len(strings)}  tokens: {len(tokens)}", flush=True)
    return labels, tokens, enc_final, lex_path, tok_path


def make_decoder(lex_path, tok_path, lm_path, beam_size, lm_weight, word_score):
    lm = lm_path if (lm_path and lm_path.lower() != "none"
                     and os.path.isfile(lm_path)) else None
    if lm is None and lm_path and lm_path.lower() != "none":
        print(f"!! WARNING: LM not found at {lm_path}; decoding WITHOUT an LM "
              f"(WER will be much worse than the paper).", flush=True)
    return ctc_decoder(
        lexicon=lex_path, tokens=tok_path, lm=lm, nbest=3,
        beam_size=beam_size, lm_weight=lm_weight, word_score=word_score,
        sil_token="|", unk_word="<unk>")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/userdata/dmoses/b3_features/zenodo")
    p.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "repro", "runs", "run1"))
    p.add_argument("--device", default="cuda")
    # language models
    p.add_argument("--lm_3gram", default=os.path.join(TEXT_DIR, "custom_lms", "full_corpus_lm_3_abs_slm.binary"))
    p.add_argument("--lm_5gram", default=os.path.join(TEXT_DIR, "custom_lms", "full_corpus_lm_5_abs_slm.binary"))
    # hyperparameters (notebook defaults / the exact run used in the paper notebook)
    p.add_argument("--decimation", type=int, default=6)
    p.add_argument("--hidden_dim", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ks", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--feat_stream", default="both", choices=["both", "hga", "raw"])
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--LM_WEIGHT", type=float, default=4.0)
    p.add_argument("--WORD_SCORE", type=float, default=-0.26)
    p.add_argument("--beam_width", type=int, default=100)
    p.add_argument("--jitter_amt", type=float, default=1.0)
    p.add_argument("--chan_noise", type=float, default=0.0)
    p.add_argument("--blackout_prob", type=float, default=0.05)
    p.add_argument("--word_ct_weight", type=float, default=0.0)
    p.add_argument("--clipamt", type=float, default=0.0001)
    p.add_argument("--winstart", type=float, default=-0.5)
    p.add_argument("--winend", type=float, default=7.5)
    p.add_argument("--normalization_strategy", default="norm_times")
    p.add_argument("--eval_set", type=int, default=1)
    p.add_argument("--train_amt", type=float, default=1.0)
    p.add_argument("--samples_to_trim", type=int, default=0)
    p.add_argument("--max_epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=10)
    # final-eval beam search
    p.add_argument("--final_beam_size", type=int, default=3000)
    p.add_argument("--final_lm_weight", type=float, default=4.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_ctc_dir = os.path.join(args.out_dir, "for_ctc")
    print("torch", torch.__version__, "torchaudio", torchaudio.__version__, flush=True)
    print("device:", args.device, "cuda avail:", torch.cuda.is_available(), flush=True)
    print("args:", json.dumps(vars(args), indent=2), flush=True)
    if args.seed:
        torch.manual_seed(args.seed)

    # --- labels ---
    labels = pd.read_hdf(os.path.join(args.data_dir, "training_labels.h5"))
    labels_te = pd.read_hdf(os.path.join(args.data_dir, "realtime_test_labels.h5"))
    for l in labels_te["txt_label"].values:
        assert l not in labels["txt_label"].values, "test sentence leaked into train!"
    labels = pd.concat((labels, labels_te), ignore_index=True)

    # --- neural data ---
    assert args.decimation == 6, "only the pre-decimated (decimation=6) data is provided"
    X = np.load(os.path.join(args.data_dir, "train_data.npy"))
    X_te = np.load(os.path.join(args.data_dir, "realtime_test_data.npy"))
    n_test = X_te.shape[0]
    X = np.concatenate((X, X_te), axis=0).astype(np.float32)
    print(f"train samples {X.shape[0] - n_test}  test samples {n_test}  X {X.shape}", flush=True)

    X = normalize_X(X, args.normalization_strategy)
    if args.feat_stream == "hga":
        X = X[:, :, :X.shape[-1] // 2]
    elif args.feat_stream == "raw":
        X = X[:, :, X.shape[-1] // 2:]
    print("final X shape", X.shape, flush=True)

    # --- lexicon / tokens / encoders ---
    labels, tokens, enc_final, lex_path, tok_path = build_lexicon_and_tokens(
        labels, out_ctc_dir)

    # --- CTC targets ---
    y_final = []
    for targ in labels["ph_label"]:
        cur = [enc_final["|"]] + [enc_final[ph] for ph in targ] + [enc_final["|"]]
        y_final.append(cur)
    maxlen = max(len(y) for y in y_final)
    Y = -1 * np.ones((len(y_final), maxlen))
    outlens = []
    for k, y in enumerate(y_final):
        Y[k, :len(y)] = np.array(y)
        outlens.append(len(y))
    outlens = np.array(outlens)

    lens = np.array([(l // args.decimation) for l in labels["length"]]) - args.samples_to_trim
    lens = np.array([min(l, X.shape[1]) for l in lens])

    gt_text = labels["txt_label"].values
    word_targets = np.array([len(l.split(" ")) for l in labels["txt_label"].values])
    n_wordtarg = int(word_targets.max())

    # --- splits (seed 1337, identical to the notebook) ---
    inds = np.arange(len(X))
    test_day_inds = copy.deepcopy(inds[-n_test:])
    off_limits = list(test_day_inds)
    test_inds_eligible = inds
    np.random.seed(1337)
    np.random.shuffle(test_inds_eligible)
    trainsets = []
    for k in range(10):
        te = test_inds_eligible[k * (len(inds) // 20):(k + 1) * (len(inds) // 20)]
        te = [i for i in te if i not in off_limits]
        val = test_inds_eligible[(k + 2) * (len(inds) // 20):(k + 3) * (len(inds) // 20)]
        val = [i for i in val if i not in te and i not in off_limits]
        tr = [i for i in inds if i not in list(te) + list(val) and i not in off_limits]
        trainsets.append((tr, val, te))
    trainsets = trainsets[args.eval_set:args.eval_set + 1]

    # --- augmentations (notebook's tuned values) ---
    b1 = {"additive_noise_level": 0.0027354917297051813,
          "scale_low": 0.9551356218945801, "scale_high": 1.0713824626558794,
          "blackout_len": 0.30682868940865543}
    train_jitter = Jitter((-1, 8), (args.winstart, args.winend),
                          jitter_amt=args.jitter_amt, decimation=args.decimation)
    test_jitter = Jitter((-1, 8), (args.winstart, args.winend),
                         jitter_amt=0.0, decimation=args.decimation)
    lens[:] = train_jitter.winsize  # every trial uses the same fixed window
    composed = transforms.Compose([
        train_jitter, Blackout(b1["blackout_len"], args.blackout_prob),
        AdditiveNoise(b1["additive_noise_level"]),
        LevelChannelNoise(args.chan_noise),
        ScaleAugment(b1["scale_low"], b1["scale_high"]),
    ])
    test_augs = transforms.Compose([test_jitter])

    # --- decoders ---
    greedy = GreedyCTCDecoder(labels=list(enc_final.keys()))
    beam_train = make_decoder(lex_path, tok_path, args.lm_3gram,
                              args.beam_width, args.LM_WEIGHT, args.WORD_SCORE)

    # --- build the single fold's datasets ---
    train, val, test = trainsets[0]
    val = val[:450]
    test = test[:450]
    train = train[:int(len(train) * args.train_amt)]
    print(f"sizes  tr/val/test  {len(train)}/{len(val)}/{len(test)}", flush=True)

    def dset(idx, tf):
        idx = np.array(idx)
        return CTCDataset_Wordct(X[idx], Y[idx], lens[idx], outlens[idx], idx,
                                 word_targets[idx], transform=tf)

    train_loader = DataLoader(dset(train, composed), batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(dset(val, test_augs),
                            batch_size=min(args.bs, len(val)), shuffle=False)

    # --- model + optimizer ---
    model = AUXCnnRnnClassifier(
        rnn_dim=args.hidden_dim, KS=args.ks, num_layers=args.num_layers,
        dropout=args.dropout, n_targ=len(enc_final), bidirectional=True,
        in_channels=X.shape[-1], nword_targ=n_wordtarg + 1).to(args.device)
    if args.weight_decay is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)

    print("\n===== TRAINING =====", flush=True)
    model, best_val_wer = train_loop(
        model, train_loader, val_loader, optimizer, args.device, gt_text,
        greedy, beam_train, tokens, args.out_dir, max_epochs=args.max_epochs,
        patience=args.patience, start_eval=0, wordct_weight=args.word_ct_weight,
        clipamt=args.clipamt)
    print(f"best validation WER (3-gram, beam {args.beam_width}): {best_val_wer:.4f}", flush=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "final_model.pth"))

    # --- final realtime evaluation with the 5-gram LM + bigger beam ---
    print("\n===== REALTIME TEST EVALUATION (unseen sentences) =====", flush=True)
    beam_final = make_decoder(lex_path, tok_path, args.lm_5gram,
                              args.final_beam_size, args.final_lm_weight,
                              args.WORD_SCORE)
    final_loader = DataLoader(dset(list(test_day_inds), test_augs),
                              batch_size=1, shuffle=False)
    res = eval_realtime_wpm(model, final_loader, args.device, greedy, beam_final,
                            gt_text, tokens)

    gt_word_counts = np.array([len(t.split()) for t in res["gts"]])
    pb_wer = parceled_metric(res["wers"], lengths=gt_word_counts)
    pb_cer = parceled_metric(res["cers"], lengths=gt_word_counts)
    pb_per = parceled_metric(res["pers"], lengths=gt_word_counts)
    median_wpm = float(np.median(res["wpms"]))

    summary = {
        "best_val_wer_3gram": best_val_wer,
        "realtime_net_wer": res["net_wer"],
        "realtime_net_cer": res["net_cer"],
        "realtime_net_per": res["net_per"],
        "realtime_median_wer": float(np.median(res["wers"])),
        "pseudoblocked_median_wer": float(np.median(pb_wer)),
        "pseudoblocked_median_cer": float(np.median(pb_cer)),
        "pseudoblocked_median_per": float(np.median(pb_per)),
        "median_wpm": median_wpm,
        "n_test_sentences": len(res["wers"]),
    }
    print("\n===== SUMMARY =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nPaper headline: median WER ~25%, ~78 WPM", flush=True)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame({
        "gt": res["gts"], "decoded": res["transcripts"],
        "wer": res["wers"], "cer": res["cers"], "per": res["pers"],
        "wpm": res["wpms"], "speak_time": res["speak_times"],
    }).to_csv(os.path.join(args.out_dir, "realtime_predictions.csv"), index=False)
    print("wrote summary.json and realtime_predictions.csv to", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
