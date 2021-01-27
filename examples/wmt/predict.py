from train import *
from absl import app
from absl import flags
from absl import logging

from clu import platform
import train
import jax
from ml_collections import config_flags
import tensorflow as tf
import jax.numpy as jnp

import functools
import tokenizer

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.DEFINE_string("jax_backend_target", None,
                    "JAX backend target to use. Can be used with UPTC.")
flags.mark_flags_as_required(["config", "workdir"])

def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    tf.config.experimental.set_visible_devices([], "GPU")

    if FLAGS.jax_backend_target:
        logging.info("Using JAX backend target %s", FLAGS.jax_backend_target)
        # jax.config.update("jax_xla_backend", "tpu_driver")
        jax.config.update("jax_backend_target", FLAGS.jax_backend_target)

    logging.info("JAX host: %d / %d", jax.host_id(), jax.host_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(
        f"host_id: {jax.host_id()}, host_count: {jax.host_count()}")
    platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                         FLAGS.workdir, "workdir")

    p_init_cache, p_pred_step, target, decode_tokens, encoder, m, predict_config = build_pred_thunks(FLAGS.config, FLAGS.workdir)
    # p_tokenizer = tokenizer.TokenizeOp(encoder)
    # p_tokenize = encoder.tokenize
    p_translate_step = functools.partial(
        decode_step,
        p_pred_step=p_pred_step,
        p_init_cache=p_init_cache,
        target=target,
        decode_tokens=decode_tokens,
        max_predict_length=FLAGS.config.max_predict_length,
    )
    print("OK")
    ragged = encoder.tokenize(["foo bar baz", "HELLO THIS IS SPONGEBOB"])
    def mk_ragged_iterator(ragged):
        for row_index in tf.range(ragged.nrows()):
            yield ragged[row_index]
    batch = []
    for tks in mk_ragged_iterator(ragged):
        padding_length = 32 - len(tks)
        if padding_length < 0:
            batch.append(tks[:padding_length])
        else:
            batch.append(tf.pad(tks, [[0, padding_length]]))
    pred_batch = tf.stack(batch)# ._numpy()
    # print("PRED BATCH: ", pred_batch)
    # print()
    # decode_target = jnp.array([[0], [0]], dtype=jnp.int32)
    # # result = target(pred_batch._numpy(), decode_target)
    # result = models.Transformer(predict_config).apply(
    #     {"params":target},
    #     pred_batch,
    #     method=models.Transformer.encode
    # )

    # print("RESULT: ", result)

    # result, cache = models.Transformer(predict_config).apply(
    #     {"params":target},
    #     result,
    #     pred_batch,
    #     decode_target,
    #     mutable=["cache"],
    #     method=models.Transformer.decode
    # )

    # print("RESULT: ", result)

    result = p_translate_step(pred_batch=pred_batch)
    print("RESULT: ", result)
    # result = m
            

if __name__ == "__main__":
    app.run(main)
