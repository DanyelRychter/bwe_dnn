"""Schritt 14 — GAN-Losses (Diskriminator-BCE, Adversarial, Feature-Matching)."""

import numpy as np
import tensorflow as tf

from bwe.losses import (
    discriminator_loss, feature_matching_loss, generator_adv_loss,
)


def test_discriminator_loss_low_when_correct():
    """Perfekter D: echt≫0, fake≪0 → Loss ~0."""
    real = tf.fill([2, 8, 4, 1], 10.0)
    fake = tf.fill([2, 8, 4, 1], -10.0)
    assert float(discriminator_loss(real, fake)) < 1e-3


def test_discriminator_loss_high_when_swapped():
    """Vertauschte Urteile → großer Loss."""
    real = tf.fill([2, 8, 4, 1], -10.0)
    fake = tf.fill([2, 8, 4, 1], 10.0)
    assert float(discriminator_loss(real, fake)) > 5.0


def test_generator_adv_loss_rewards_fooling():
    """G täuscht D (fake≫0) → Loss ~0; durchschaut (fake≪0) → groß."""
    assert float(generator_adv_loss(tf.fill([2, 8, 4, 1], 10.0))) < 1e-3
    assert float(generator_adv_loss(tf.fill([2, 8, 4, 1], -10.0))) > 5.0


def test_feature_matching_zero_when_identical():
    feats = [tf.random.normal([2, 16, 8, c]) for c in (64, 128)]
    assert float(feature_matching_loss(feats, feats)) == 0.0


def test_feature_matching_positive_and_differentiable():
    """>0 bei Unterschied; Gradient fließt in die fake-Aktivierungen."""
    real = [tf.random.normal([2, 16, 8, 64])]
    fake = [tf.Variable(tf.random.normal([2, 16, 8, 64]))]
    with tf.GradientTape() as tape:
        loss = feature_matching_loss(real, fake)
    assert float(loss) > 0.0
    g = tape.gradient(loss, fake[0])
    assert g is not None and np.abs(g.numpy()).sum() > 0.0
