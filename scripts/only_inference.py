import glob
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.stats import pearsonr
from tensorflow.keras.models import load_model

# ----------------------- CONFIG -----------------------
MAT_FOLDER = "."                  # cartella con i .mat
MAT_VAR = "Tab_in_NN"             # variabile dentro ai .mat
MODEL_PATH = "cnn_hybrid_only_inference.h5"
SCALER_X_PATH = "scaler_x.save"
SCALER_Y_PATH = "scaler_y.save"

INPUT_COLS = [0, 5, 6]             # colonne input
OUTPUT_COLS = [2, 3, 4]            # colonne output
TARGET_LEN = 100                   # lunghezza di riscampionamento (come nel training)

APPLY_LAG_PER_SEQUENCE = True      # opzionale: correzione di fase per analisi offline
MAX_LAG = 3

DO_PLOTS = True                    # plot di qualche stance e della media


# ----------------------- I/O E PREP -------------------
def load_and_segment_all_patients_from_folder(folder=".", mat_variable_name="Tab_in_NN"):
    all_stances, all_patient_ids = [], []
    mat_files = sorted(glob.glob(f"{folder}/*.mat"))
    if len(mat_files) == 0:
        raise FileNotFoundError("Nessun file .mat trovato nella cartella specificata.")

    for patient_idx, file_path in enumerate(mat_files):
        mat = scipy.io.loadmat(file_path)
        if mat_variable_name not in mat:
            raise KeyError(f"Variabile '{mat_variable_name}' non trovata in {file_path}.")

        matrix = mat[mat_variable_name]
        stances = segment_stances(matrix)
        all_stances.extend(stances)
        all_patient_ids.extend([patient_idx] * len(stances))

    if len(all_stances) == 0:
        raise ValueError("Nessuna stance segmentata: controlla i dati e segment_stances().")

    return all_stances, np.array(all_patient_ids, dtype=int)


def segment_stances(matrix):
    """
    Segmentazione usando la colonna 1 (2a colonna) come marker di reset.

    Robustezza aggiunta:
    - Se non ci sono reset espliciti, considera tutta la matrice come una stance.
    - Mantiene anche stance di lunghezza 1 (caso richiesto: inferenza a un solo passo).
    """
    reset_indices = np.where(matrix[:, 1] == 1)[0]

    if reset_indices.size == 0:
        return [matrix.copy()] if len(matrix) > 0 else []

    stances = []
    for i, start in enumerate(reset_indices):
        end = reset_indices[i + 1] if i + 1 < len(reset_indices) else len(matrix)
        stance = matrix[start:end]
        if len(stance) > 0:
            stances.append(stance.copy())
    return stances


def resample_sequence(seq, target_len=100):
    """
    Riscampiona una sequenza a target_len.

    Fix principale per sequenze di 1 solo passo:
    - Se original_len == 1, replica il singolo campione target_len volte.
    """
    original_len, n_features = seq.shape
    if original_len == 0:
        raise ValueError("Sequenza vuota: impossibile riscampionare.")

    if original_len == 1:
        return np.repeat(seq, target_len, axis=0)

    new_indices = np.linspace(0, original_len - 1, target_len)
    resampled = np.zeros((target_len, n_features), dtype=seq.dtype)
    base_idx = np.arange(original_len)
    for i in range(n_features):
        resampled[:, i] = np.interp(new_indices, base_idx, seq[:, i])
    return resampled


def prepare_sequences(stances, patient_ids, input_cols, output_cols, target_len):
    X_list, y_list, pid_list = [], [], []
    for stance, pid in zip(stances, patient_ids):
        X = stance[:, input_cols]
        y = stance[:, output_cols]
        X_list.append(resample_sequence(X, target_len))
        y_list.append(resample_sequence(y, target_len))
        pid_list.append(pid)
    return np.array(X_list), np.array(y_list), np.array(pid_list)


def inverse_normalize_y(y_norm, scaler_y):
    orig_shape = y_norm.shape
    y_flat = y_norm.reshape(-1, orig_shape[2])
    y_orig = scaler_y.inverse_transform(y_flat)
    return y_orig.reshape(orig_shape)


# ----------------------- METRICHE ---------------------
def _rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def _rrmse_peak2peak(a, b):
    rmse = _rmse(a, b)
    denom = 0.5 * ((np.max(a) - np.min(a)) + (np.max(b) - np.min(b)))
    return np.nan if denom <= 0 else (rmse / denom) * 100.0


def _nrmse_range_gt(a, b):
    rmse = _rmse(a, b)
    rng = np.max(a) - np.min(a)  # solo ground truth
    return np.nan if rng <= 0 else (rmse / rng) * 100.0


def _pearson(a, b):
    if a.size < 2 or np.all(a == a.flat[0]) or np.all(b == b.flat[0]):
        return np.nan
    return pearsonr(a, b)[0]


def metrics_per_sequence(y_true_seq, y_pred_seq):
    _, C = y_true_seq.shape
    rmse_c, rrmse_c, nrmse_c, corr_c = [], [], [], []
    for c in range(C):
        yt = y_true_seq[:, c]
        yp = y_pred_seq[:, c]
        rmse_c.append(_rmse(yt, yp))
        rrmse_c.append(_rrmse_peak2peak(yt, yp))
        nrmse_c.append(_nrmse_range_gt(yt, yp))
        corr_c.append(_pearson(yt, yp))
    return rmse_c, rrmse_c, nrmse_c, corr_c


def metrics_across_sequences(y_true, y_pred, reduce="mean"):
    N, _, C = y_true.shape
    all_rmse = np.zeros((N, C))
    all_rrmse = np.zeros((N, C))
    all_nrmse = np.zeros((N, C))
    all_corr = np.zeros((N, C))

    for i in range(N):
        r, rr, nr, cc = metrics_per_sequence(y_true[i], y_pred[i])
        all_rmse[i] = r
        all_rrmse[i] = rr
        all_nrmse[i] = nr
        all_corr[i] = cc

    agg = np.nanmean if reduce == "mean" else np.nanmedian
    return {
        "rmse_mean": agg(all_rmse, axis=0),
        "rmse_std": np.nanstd(all_rmse, axis=0),
        "rrmse_mean": agg(all_rrmse, axis=0),
        "rrmse_std": np.nanstd(all_rrmse, axis=0),
        "nrmse_mean": agg(all_nrmse, axis=0),
        "nrmse_std": np.nanstd(all_nrmse, axis=0),
        "corr_mean": agg(all_corr, axis=0),
        "corr_std": np.nanstd(all_corr, axis=0),
    }


def metrics_per_patient(y_true, y_pred, patient_ids):
    patients = np.unique(patient_ids)
    per_patient = []
    for pid in patients:
        sel = patient_ids == pid
        per_patient.append(metrics_across_sequences(y_true[sel], y_pred[sel]))

    keys = ["rmse_mean", "rrmse_mean", "nrmse_mean", "corr_mean"]
    agg = {}
    for k in keys:
        M = np.stack([d[k] for d in per_patient], axis=0)
        agg[k] = np.nanmean(M, axis=0)
        agg[k.replace("_mean", "_std")] = np.nanstd(M, axis=0)
    return agg


def pretty_print_metrics(title, comp_names, stats):
    print(f"\n{title}")
    for i, cname in enumerate(comp_names):
        print(
            f"  {cname}: RMSE={stats['rmse_mean'][i]:.4f}±{stats['rmse_std'][i]:.4f} | "
            f"rRMSE%={stats['rrmse_mean'][i]:.2f}±{stats['rrmse_std'][i]:.2f} | "
            f"NRMSE_range%={stats['nrmse_mean'][i]:.2f}±{stats['nrmse_std'][i]:.2f} | "
            f"Corr={stats['corr_mean'][i]:.4f}±{stats['corr_std'][i]:.4f}"
        )


# ------------------- LIGHT POST-PROCESSING (opz.) ----
def best_lag_1d(a, b, max_lag=3):
    if a.size < 2 or b.size < 2:
        return 0

    a = (a - np.mean(a)) / (np.std(a) + 1e-8)
    b = (b - np.mean(b)) / (np.std(b) + 1e-8)
    lags = range(-max_lag, max_lag + 1)
    scores = []
    for lag in lags:
        scores.append(np.mean(a * np.roll(b, lag)))
    return lags[int(np.argmax(scores))]


def compute_lags_per_sequence(y_true, y_pred, max_lag=3):
    N, _, C = y_true.shape
    lags_nc = np.zeros((N, C), dtype=int)
    for i in range(N):
        for c in range(C):
            lags_nc[i, c] = best_lag_1d(y_true[i, :, c], y_pred[i, :, c], max_lag=max_lag)
    return lags_nc


def apply_lag_per_sequence(y_pred, lags_nc):
    yp = y_pred.copy()
    N, _, C = yp.shape
    for i in range(N):
        for c in range(C):
            lag = int(lags_nc[i, c])
            if lag != 0:
                yp[i, :, c] = np.roll(yp[i, :, c], lag)
    return yp


# ----------------------- PLOT (opzionale) -------------
def plot_stance_components_with_inputs(
    X,
    y_true,
    y_pred,
    stance_idxs,
    title_prefix="Only-Inference ",
    input_names=None,
    output_names=None,
):
    n_in = X.shape[2]
    n_out = y_true.shape[2]

    if input_names is None:
        input_names = ["R", "Cop_ML", "Cop_AP"]
    if output_names is None:
        output_names = ["ML", "V", "AP"]

    def adapt_labels(names, n, fallback_prefix):
        if len(names) >= n:
            return names[:n]
        return names + [f"{fallback_prefix} {i}" for i in range(len(names), n)]

    input_labels = adapt_labels(input_names, n_in, "Input")
    output_labels = adapt_labels(output_names, n_out, "Output")

    for idx in stance_idxs:
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        for i in range(n_in):
            axs[0].plot(X[idx, :, i], label=input_labels[i])
        axs[0].set_title(f"{title_prefix}Stance {idx} - Input")
        axs[0].legend()
        axs[0].grid(True)

        for i in range(n_out):
            axs[1].plot(y_true[idx, :, i], label=f"True {output_labels[i]}", linestyle="-")
            axs[1].plot(y_pred[idx, :, i], label=f"Pred {output_labels[i]}", linestyle="--")
        axs[1].set_title(f"{title_prefix}Stance {idx} - True vs Pred")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()


def plot_mean_step_with_metrics(
    y_true,
    y_pred,
    comp_names=("ML", "V", "AP"),
    title="Only-Inference - Mean stance + band RMSE(t)",
):
    _, T, C = y_true.shape
    t = np.arange(T)
    fig, axs = plt.subplots(C, 1, figsize=(12, 9), sharex=True)
    if C == 1:
        axs = [axs]

    for c in range(C):
        mu_true = np.nanmean(y_true[:, :, c], axis=0)
        mu_pred = np.nanmean(y_pred[:, :, c], axis=0)
        err = y_pred[:, :, c] - y_true[:, :, c]
        rmse_t = np.sqrt(np.nanmean(err ** 2, axis=0))

        ax = axs[c]
        ax.plot(t, mu_true, label=f"True {comp_names[c]}")
        ax.plot(t, mu_pred, linestyle="--", label=f"Pred {comp_names[c]}")
        ax.fill_between(t, mu_pred - rmse_t, mu_pred + rmse_t, alpha=0.25, label="± RMSE(t)")
        ax.set_ylabel(comp_names[c])
        ax.grid(True)
        ax.legend(loc="best")

    axs[-1].set_xlabel("Stance cycle")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ----------------------- MODEL LOADER -----------------
def load_frozen_model_for_inference(model_path):
    model = load_model(model_path, compile=False)
    for layer in model.layers:
        layer.trainable = False
    return model


# ----------------------- PIPELINE ---------------------
def run_only_inference():
    # 1) Carica e prepara dati nuovi
    new_stances, new_patient_ids = load_and_segment_all_patients_from_folder(
        folder=MAT_FOLDER,
        mat_variable_name=MAT_VAR,
    )
    new_X, new_y, new_pid_seq = prepare_sequences(
        new_stances,
        new_patient_ids,
        input_cols=INPUT_COLS,
        output_cols=OUTPUT_COLS,
        target_len=TARGET_LEN,
    )
    assert new_X.shape[1] == TARGET_LEN, f"target_len deve essere {TARGET_LEN} come nel training"

    # 2) Carica scaler e modello
    scaler_x = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    model = load_frozen_model_for_inference(MODEL_PATH)

    # 3) Normalizza X come in training, predici e de-normalizza y
    Xn = scaler_x.transform(new_X.reshape(-1, new_X.shape[2])).reshape(new_X.shape)
    y_pred_norm = model.predict(Xn, verbose=1)
    y_pred = inverse_normalize_y(y_pred_norm, scaler_y)

    comp_names = ("ML", "V", "AP")

    print("\n=== BASELINE (senza correzione di lag) ===")
    s_seq_base = metrics_across_sequences(new_y, y_pred)
    s_pat_base = metrics_per_patient(new_y, y_pred, new_pid_seq)
    pretty_print_metrics("Baseline (per-sequence mean±std)", comp_names, s_seq_base)
    pretty_print_metrics("Baseline (per-paziente mean±std)", comp_names, s_pat_base)

    # 4) (Opz.) lag per-sequenza per sola analisi
    y_pred_used = y_pred
    if APPLY_LAG_PER_SEQUENCE:
        print("\n=== APPLYING PER-SEQUENCE LAG CORRECTION ===")
        lags_nc = compute_lags_per_sequence(new_y, y_pred, max_lag=MAX_LAG)
        y_pred_used = apply_lag_per_sequence(y_pred, lags_nc)

        print("\n=== METRICHE DOPO CORREZIONE DI LAG ===")
        s_seq_lag = metrics_across_sequences(new_y, y_pred_used)
        s_pat_lag = metrics_per_patient(new_y, y_pred_used, new_pid_seq)
        pretty_print_metrics("Post-Lag (per-sequence mean±std)", comp_names, s_seq_lag)
        pretty_print_metrics("Post-Lag (per-paziente mean±std)", comp_names, s_pat_lag)

        print("\nΔMETRICHE POST-LAG (miglioramento; +Corr, -RMSE sono buoni):")
        for i, cname in enumerate(comp_names):
            diff_corr = s_seq_lag["corr_mean"][i] - s_seq_base["corr_mean"][i]
            diff_rmse = s_seq_base["rmse_mean"][i] - s_seq_lag["rmse_mean"][i]
            print(f"  {cname}: ΔCorr=+{diff_corr:.3f} | ΔRMSE=-{diff_rmse:.4f}")

    # 5) Metriche finali
    s_seq = metrics_across_sequences(new_y, y_pred_used)
    s_pat = metrics_per_patient(new_y, y_pred_used, new_pid_seq)
    pretty_print_metrics("Only-Inference (per-sequence mean±std)", comp_names, s_seq)
    pretty_print_metrics("Only-Inference (per-paziente mean±std)", comp_names, s_pat)

    # 6) Plot (opzionali)
    if DO_PLOTS and len(new_y) > 0:
        stance_idxs = random.sample(range(len(new_y)), min(6, len(new_y)))
        plot_stance_components_with_inputs(new_X, new_y, y_pred_used, stance_idxs)
        plot_mean_step_with_metrics(new_y, y_pred_used, comp_names=comp_names)

    return y_pred_used, {"seq": s_seq, "patient": s_pat}


if __name__ == "__main__":
    run_only_inference()
