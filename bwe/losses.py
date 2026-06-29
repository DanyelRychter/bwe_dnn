"""Loss + Splicing für die komplexe Regression (Stufe 1) — und später das GAN.

* :func:`splice` — LF bit-genau aus dem Input, HF aus der Modellvorhersage
  (differenzierbar). Gehört untrennbar zum HF-only-Loss: ohne Splicing bekäme das
  Modell kein Signal, das LF zu erhalten. Auch im GAN (vor dem Diskriminator) und in
  der Inferenz verwendet.
* :func:`ri_mag_loss` — Summe dreier L1-Terme (Real, Imag, Magnitude), **nur auf den
  HF-Bins**. (1)+(2) erfassen die Phase (steckt in der Relation Re↔Im); (3) zwingt die
  energetisch korrekte Magnitude (die das Ohr am stärksten wahrnimmt).

Spektren sind ``[B, F, T, C]`` mit C≥2 (Kanäle 0,1 = Re, Im; weitere ignoriert).
"""

from __future__ import annotations

import tensorflow as tf

from bwe import config as cfg


def _hf_mask_f(n_bins, cutoff_bin, dtype):
    """[F]-Maske: 1.0 für HF-Bins (>= cutoff_bin), sonst 0.0."""
    return tf.cast(tf.range(n_bins) >= cutoff_bin, dtype)


def splice(model_out, input_spec, cutoff_bin: int = cfg.CUTOFF_BIN):
    """Vollspektrogramm: LF aus ``input_spec[...,:2]``, HF aus ``model_out`` (beide ``[B,F,T,2]``)."""
    model_out = tf.convert_to_tensor(model_out)
    lf = input_spec[..., : cfg.N_OUTPUT_CHANNELS]
    m = _hf_mask_f(tf.shape(model_out)[1], cutoff_bin, model_out.dtype)
    m = m[tf.newaxis, :, tf.newaxis, tf.newaxis]                 # [1,F,1,1]
    return model_out * m + tf.cast(lf, model_out.dtype) * (1.0 - m)


def ri_mag_loss(
    pred, target,
    cutoff_bin: int = cfg.CUTOFF_BIN,
    w_re: float = cfg.W_RE, w_im: float = cfg.W_IM, w_mag: float = cfg.W_MAG,
    eps: float = cfg.LOSS_EPS,
):
    """L1(Re)+L1(Im)+w_mag·L1(|·|) der HF-Bins. ``pred``/``target``: ``[B,F,T,2]``."""
    pred = tf.convert_to_tensor(pred)
    target = tf.cast(target, pred.dtype)

    m = _hf_mask_f(tf.shape(pred)[1], cutoff_bin, pred.dtype)[tf.newaxis, :, tf.newaxis]  # [1,F,1]

    def masked_l1(a, b):                                          # a,b: [B,F,T]
        diff = tf.abs(a - b) * m
        denom = tf.reduce_sum(m) * tf.cast(tf.shape(a)[0] * tf.shape(a)[2], pred.dtype)
        return tf.reduce_sum(diff) / (denom + eps)

    re_p, im_p = pred[..., 0], pred[..., 1]
    re_t, im_t = target[..., 0], target[..., 1]
    mag_p = tf.sqrt(tf.square(re_p) + tf.square(im_p) + eps)
    mag_t = tf.sqrt(tf.square(re_t) + tf.square(im_t) + eps)

    return (w_re * masked_l1(re_p, re_t)
            + w_im * masked_l1(im_p, im_t)
            + w_mag * masked_l1(mag_p, mag_t))


# --------------------------------------------------------------------------- #
# GAN-Terme (Stufe 2) — Vanilla-BCE auf den Patch-Logits (Spectral Norm → kein
# Sigmoid im Diskriminator, daher ``from_logits``).
# --------------------------------------------------------------------------- #
def _bce_logits(labels, logits):
    """Gemittelte Binary-Cross-Entropy direkt auf Logits (numerisch stabil)."""
    logits = tf.convert_to_tensor(logits)
    labels = tf.cast(labels, logits.dtype)
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    )


def discriminator_loss(real_logits, fake_logits):
    """``bce(1, echt) + bce(0, fake)`` — der Diskriminator trennt echtes von generiertem HF."""
    return (_bce_logits(tf.ones_like(real_logits), real_logits)
            + _bce_logits(tf.zeros_like(fake_logits), fake_logits))


def generator_adv_loss(fake_logits):
    """``bce(1, fake)`` — der Generator will, dass der Diskriminator „echt" sagt."""
    return _bce_logits(tf.ones_like(fake_logits), fake_logits)


def feature_matching_loss(feats_real, feats_fake):
    """Mittlerer L1-Abstand der Diskriminator-Zwischenaktivierungen (echt vs. fake).

    Stabilisiert das Generator-Training: zieht die Statistik der generierten Ausgabe
    schichtweise an die echte heran, ohne den (zickigen) reinen Adversarial-Gradienten.
    ``feats_real``/``feats_fake``: gleich lange Listen von Aktivierungstensoren.
    """
    terms = [tf.reduce_mean(tf.abs(tf.cast(r, f.dtype) - f))
             for r, f in zip(feats_real, feats_fake)]
    return tf.add_n(terms) / float(len(terms))
