"""
Evaluate a trained text-decoder checkpoint and dump a metrics pickle.

Produces the same metric structure as the lab's eval script: both a "raw"
evaluation (full 7.5 s window, no early stopping) and an early-stopping
evaluation (the real-time protocol), saved to a .pkl for plotting.

It reuses the EXACT data prep, model, decoders and pseudo-blocking from
train_text_decoder.py so evaluation is consistent with training.

Pickle contents:
    model_weights_path
    raw_wers, raw_greedy_cers, raw_cers, raw_pers   (full-window eval)
    wers, cers, pers                                (early-stopping eval)
    wpms, speaking_times, early_stopping            (early-stopping eval)
    gts, transcripts
    net_wer, pseudoblocked_median_wer, median_wpm   (aggregates)

Example (evaluate one run's checkpoint with the 5-gram LM):
    python repro/eval_model.py \
        repro/runs/bs_64/final_model.pth \
        repro/runs/bs_64/metrics.pkl \
        --data_dir /userdata/dmoses/b3_features/zenodo
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# reuse everything from the training script (also sets up sys.path to text/)
import train_text_decoder as T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_test(args):
    """Rebuild X / labels / tokens / test split exactly like training."""
    labels = pd.read_hdf(os.path.join(args.data_dir, "training_labels.h5"))
    labels_te = pd.read_hdf(os.path.join(args.data_dir, "realtime_test_labels.h5"))
    labels = pd.concat((labels, labels_te), ignore_index=True)

    X = np.load(os.path.join(args.data_dir, "train_data.npy")).astype(np.float32)
    X_te = np.load(os.path.join(args.data_dir, "realtime_test_data.npy")).astype(np.float32)
    n_test = X_te.shape[0]
    X = np.concatenate((X, X_te), axis=0)
    X = T.normalize_X(X, args.normalization_strategy)
    if args.feat_stream == "hga":
        X = X[:, :, :X.shape[-1] // 2]
    elif args.feat_stream == "raw":
        X = X[:, :, X.shape[-1] // 2:]

    out_ctc_dir = os.path.join(args.out_dir, "for_ctc")
    labels, tokens, enc_final, lex_path, tok_path = T.build_lexicon_and_tokens(labels, out_ctc_dir)

    y_final = []
    for targ in labels["ph_label"]:
        y_final.append([enc_final["|"]] + [enc_final[ph] for ph in targ] + [enc_final["|"]])
    maxlen = max(len(y) for y in y_final)
    Y = -1 * np.ones((len(y_final), maxlen))
    outlens = []
    for k, y in enumerate(y_final):
        Y[k, :len(y)] = np.array(y)
        outlens.append(len(y))
    outlens = np.array(outlens)

    lens = np.array([(l // args.decimation) for l in labels["length"]])
    lens = np.array([min(l, X.shape[1]) for l in lens])

    gt_text = labels["txt_label"].values
    word_targets = np.array([len(l.split(" ")) for l in labels["txt_label"].values])
    n_wordtarg = int(word_targets.max())

    test_day_inds = np.arange(len(X))[-n_test:]
    test_jitter = T.Jitter((-1, 8), (args.winstart, args.winend),
                           jitter_amt=0.0, decimation=args.decimation)
    lens[:] = test_jitter.winsize
    test_augs = T.Compose([test_jitter])

    return dict(X=X, Y=Y, lens=lens, outlens=outlens, tokens=tokens, enc_final=enc_final,
                gt_text=gt_text, word_targets=word_targets, n_wordtarg=n_wordtarg,
                test_day_inds=test_day_inds, test_augs=test_augs,
                lex_path=lex_path, tok_path=tok_path)


@torch.no_grad()
def raw_eval(model, loader, greedy, beam, texts, tokens):
    """Full-window decode (no early stopping)."""
    model.eval()
    raw_wers, raw_greedy_pers, raw_cers, raw_pers = [], [], [], []
    for x, y, l, targ_len, gtsent, _ in loader:
        x = x.float().to(DEVICE)
        y = y.long().to(DEVICE)
        l = l.int().to(DEVICE)
        gtsent = gtsent.long().cpu().numpy()
        emissions, _, _ = model(x, l)
        emissions = F.log_softmax(emissions, dim=-1)
        for k, e in enumerate(emissions.permute(1, 0, 2)):
            gt_phones = [tokens[yy] for yy in y.detach().cpu().numpy()[k]
                         if yy != -1 and yy != 0]
            gt = texts[int(gtsent[k])]
            gper, _, _, cer, wer, per = T.greedy_beam_wer_cer(e, gt, gt_phones, greedy, beam)
            raw_wers.append(wer)
            raw_greedy_pers.append(gper)
            raw_cers.append(cer)
            raw_pers.append(per)
    return raw_wers, raw_greedy_pers, raw_cers, raw_pers


@torch.no_grad()
def earlystop_eval(model, loader, greedy, beam, texts, tokens):
    """Real-time protocol: decode growing windows, stop on silence detection."""
    model.eval()
    wers, cers, pers, wpms, speak_times, early_flags = [], [], [], [], [], []
    gts, transcripts = [], []
    for x, y, l, targ_len, gtsent, _ in loader:
        assert x.shape[0] == 1, "early-stop eval needs batch_size=1"
        x = x.float().to(DEVICE)
        y = y.long().to(DEVICE)
        gtsent = gtsent.long().cpu().numpy()

        t_elapsed = 7.5
        emissions = None
        for t_elapsed in [1.9, 2.7, 3.5, 4.3, 5.1, 5.9, 6.7, 7.5]:
            sample_ct = int(((t_elapsed + 0.5) * 200) / 6)
            l_cur = torch.tensor([sample_ct]).int().to(DEVICE)
            emissions, _, _ = model(x[:, :sample_ct], l_cur)
            emissions = F.softmax(emissions, dim=-1)
            if T.check_1024_done(emissions.squeeze()):
                break
        # match eval_realtime_wpm: a true early stop is one that fired before the
        # last (7.5 s) window; only then is the trailing silence subtracted.
        early = t_elapsed < 7.5
        if early:
            silence_time = (8 * model.ks * 6) / 200
            speaking_time = t_elapsed - silence_time
        else:
            speaking_time = t_elapsed
        early_flags.append(int(early))
        speak_times.append(speaking_time)

        emissions = torch.log(emissions)
        for k, e in enumerate(emissions.permute(1, 0, 2)):
            gt_phones = [tokens[yy] for yy in y.detach().cpu().numpy()[k]
                         if yy != -1 and yy != 0]
            gt = texts[int(gtsent[k])]
            gper, gt_, transcript, cer, wer, per = T.greedy_beam_wer_cer(e, gt, gt_phones, greedy, beam)
            gts.append(gt_)
            transcripts.append(transcript)
            n_words = len(transcript.strip().split(" ")) if transcript.strip() else 0
            wpms.append(60 * n_words / speaking_time if speaking_time > 0 else 0.0)
            wers.append(wer)
            cers.append(cer)
            pers.append(per)
    return dict(wers=wers, cers=cers, pers=pers, wpms=wpms, speaking_times=speak_times,
                early_stopping=early_flags, gts=gts, transcripts=transcripts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_weights_path", help="path to a saved .pth checkpoint")
    p.add_argument("metrics_output_path", help="where to write the metrics .pkl")
    p.add_argument("--data_dir", default="/userdata/dmoses/b3_features/zenodo")
    p.add_argument("--out_dir", default=None, help="scratch dir for the for_ctc files")
    p.add_argument("--lm_5gram", default=os.path.join(T.TEXT_DIR, "custom_lms", "full_corpus_lm_5_abs_slm.binary"))
    p.add_argument("--beam_size", type=int, default=3000)
    p.add_argument("--lm_weight", type=float, default=4.5)
    p.add_argument("--word_score", type=float, default=-0.26)
    # architecture (must match the checkpoint's training config)
    p.add_argument("--hidden_dim", type=int, default=500)
    p.add_argument("--ks", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--feat_stream", default="both", choices=["both", "hga", "raw"])
    p.add_argument("--normalization_strategy", default="norm_times")
    p.add_argument("--decimation", type=int, default=6)
    p.add_argument("--winstart", type=float, default=-0.5)
    p.add_argument("--winend", type=float, default=7.5)
    p.add_argument("--skip_raw", action="store_true", help="skip the full-window eval")
    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.dirname(os.path.abspath(args.metrics_output_path)) or "."
    os.makedirs(args.out_dir, exist_ok=True)
    print("device:", DEVICE, flush=True)

    d = prepare_test(args)
    model = T.AUXCnnRnnClassifier(
        rnn_dim=args.hidden_dim, KS=args.ks, num_layers=args.num_layers,
        dropout=args.dropout, n_targ=len(d["enc_final"]), bidirectional=True,
        in_channels=d["X"].shape[-1], nword_targ=d["n_wordtarg"] + 1)
    state = torch.load(args.model_weights_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)

    greedy = T.GreedyCTCDecoder(labels=list(d["enc_final"].keys()))
    beam = T.make_decoder(d["lex_path"], d["tok_path"], args.lm_5gram,
                          args.beam_size, args.lm_weight, args.word_score)

    def dset(idx):
        idx = np.array(idx)
        return T.CTCDataset_Wordct(d["X"][idx], d["Y"][idx], d["lens"][idx],
                                   d["outlens"][idx], idx, d["word_targets"][idx],
                                   transform=d["test_augs"])
    loader = DataLoader(dset(list(d["test_day_inds"])), batch_size=1, shuffle=False)

    out = {"model_weights_path": args.model_weights_path}
    if not args.skip_raw:
        print("raw (full-window) evaluation ...", flush=True)
        rw, rg, rc, rp = raw_eval(model, loader, greedy, beam, d["gt_text"], d["tokens"])
        out.update(raw_wers=rw, raw_greedy_cers=rg, raw_cers=rc, raw_pers=rp)

    print("early-stopping evaluation ...", flush=True)
    es = earlystop_eval(model, loader, greedy, beam, d["gt_text"], d["tokens"])
    out.update(wers=es["wers"], cers=es["cers"], pers=es["pers"], wpms=es["wpms"],
               speaking_times=es["speaking_times"], early_stopping=es["early_stopping"],
               gts=es["gts"], transcripts=es["transcripts"])

    wc = np.array([len(g.split()) for g in es["gts"]])
    out["net_wer"] = float(np.mean(es["wers"]))
    out["pseudoblocked_median_wer"] = float(np.median(T.parceled_metric(es["wers"], wc)))
    out["median_wpm"] = float(np.median(es["wpms"]))
    out["frac_early_stopped"] = float(np.mean(es["early_stopping"]))

    with open(args.metrics_output_path, "wb") as f:
        pickle.dump(out, f)
    print(f"\nwrote {args.metrics_output_path}", flush=True)
    print(f"  net WER {out['net_wer']:.4f}  pseudoblocked median WER "
          f"{out['pseudoblocked_median_wer']:.4f}  median WPM {out['median_wpm']:.2f}  "
          f"early-stopped {out['frac_early_stopped']*100:.0f}%", flush=True)


if __name__ == "__main__":
    main()
