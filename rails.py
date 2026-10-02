"""
Generic guard for batch AI-function work on Databricks (Spark Connect / serverless).

The guard knows nothing about ai_classify. You hand it:
  * a source DataFrame with a unique id column
  * a target Delta table
  * a `transform`: DataFrame -> DataFrame, which runs the AI function on one chunk

It then provides:
  * chunking, so one hang costs one chunk, not the table
  * a hard per-chunk timeout that actually cancels the server-side query
    (interruptTag on Spark Connect, job-group cancel on classic sessions)
  * a whole-run deadline (max_run_seconds), so many slow-but-not-hung chunks
    cannot run for hours
  * a per-row input size limit, and a cap on the TOTAL input volume of a run
  * atomic per-chunk Delta appends, and automatic resume (rows that finished
    without an error are skipped, errored rows are retried on the next run)
  * circuit breakers: consecutive chunk failures, per-chunk error rate, and a
    hard cap on pending rows
  * an on_alert hook that fires on ANY abort, with a structured payload

Transform contract:
  Input : a DataFrame of source rows for one chunk (includes `id_col`).
  Output: a DataFrame that includes `id_col` and a nullable string column
          `error_message` (null means the row succeeded), plus any result columns.
  The guard adds `processed_at` and `batch_id`, and when the input limit is on,
  `input_truncated` and `skip_reason`.

Input limit (GuardConfig.max_input_chars, applied to GuardConfig.input_col):
  oversize_policy="truncate": keep the first max_input_chars characters, send
      that, and flag the row with input_truncated = true.
  oversize_policy="skip": do not send the row to the model at all. It is written
      to the target with skip_reason = 'input_too_long' and a null error_message,
      so it counts as done and is not retried every run.
  oversize_policy="fail": a pre-flight scan of the source raises before any
      model call if any row is over the limit.
  The limit is in characters, which is a cheap proxy for tokens. Pick a number
  from the length distribution the pre-flight log prints, not from a guess.
  input_col must be the same column your transform reads its text from.

Total input cap (GuardConfig.max_total_input_chars):
  Before any model call, the guard sums the characters that WILL be sent for the
  pending rows (after truncation / skipping) and refuses to start if that is over
  the cap. max_rows alone does not bound cost: 100k rows of 1k chars and 100k rows
  of 100k chars are very different bills.

Alerting (GuardConfig.on_alert):
  Called with a dict on any abort: run deadline, circuit breaker, failed cancel,
  cap exceeded, bad config, or an unexpected error. The exception is ALWAYS
  re-raised afterwards. In a Databricks job, let it propagate: the task fails and
  the job's failure notifications fire. Do not catch it and carry on.
  A hook that itself raises is logged and ignored, so it can never mask the
  original error.

Worst-case bound:
  GuardConfig.worst_case_failure_seconds is the most time the guard can spend on
  failing chunks before the consecutive-failure breaker trips. Healthy-but-slow
  runs are bounded by max_run_seconds instead.

Example:

    from ai_guard import GuardConfig, run_guarded
    from transforms import classify

    src = spark.table("cat.sch.claims").selectExpr("claim_id AS id", "note_text AS text")

    summary = run_guarded(
        spark,
        source_df=src,
        target_table="cat.sch.claims_classified",
        transform=classify("text", {"liability": "Third party blames insured",
                                    "property": "Damage to property",
                                    "other": "Anything else"}),
        cfg=GuardConfig(
            chunk_size=5000,
            timeout_s=300,
            max_run_seconds=3 * 3600,
            max_rows=100_000,
            input_col="text",
            max_input_chars=20_000,
            max_total_input_chars=500_000_000,
            oversize_policy="truncate",
            on_alert=lambda event: post_to_slack(event),   # your function
        ),
    )

Requires a Spark Connect session (serverless compute or Databricks Connect) for
tag-based cancellation. Classic sessions work through the job-group fallback.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Literal

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

log = logging.getLogger("ai_guard")

Transform = Callable[[DataFrame], DataFrame]
OversizePolicy = Literal["truncate", "skip", "fail"]


class ChunkTimeout(Exception):
    """A chunk exceeded its timeout and was cancelled."""


class RunAborted(Exception):
    """The whole run was stopped (circuit breaker, deadline, or a cancel that did not take)."""


@dataclass(frozen=True)
class GuardConfig:
    chunk_size: int = 5000
    timeout_s: int = 300
    cancel_grace_s: int = 30
    max_rows: int = 100_000
    max_retries: int = 1
    retry_backoff_s: int = 10
    max_consecutive_failures: int = 2
    max_error_rate: float = 0.05
    # Whole-run wall-clock deadline. None means no deadline (not recommended in production).
    max_run_seconds: int | None = None
    # Per-row input limit. Off unless max_input_chars is set.
    input_col: str | None = None
    max_input_chars: int | None = None
    oversize_policy: OversizePolicy = "truncate"
    # Cap on the total characters sent in one run. Needs input_col.
    max_total_input_chars: int | None = None
    # Log a warning when a chunk takes more than this fraction of timeout_s.
    slow_chunk_fraction: float = 0.5
    # Called with a dict on any abort. See module docstring.
    on_alert: Callable[[dict], None] | None = None
    # Optional server-side backstop. Documented for Databricks SQL; may not apply
    # on serverless notebook compute, so a failure to set it is logged, not fatal.
    statement_timeout_s: int | None = None

    @property
    def worst_case_failure_seconds(self) -> int:
        """Most time spent on failing chunks before the consecutive-failure breaker trips."""
        backoff = sum(self.retry_backoff_s * (a + 1) for a in range(self.max_retries))
        per_chunk = (self.max_retries + 1) * (self.timeout_s + self.cancel_grace_s) + backoff
        return per_chunk * self.max_consecutive_failures


def _classic_context(spark: SparkSession):
    """The SparkContext on a classic session, or None on Spark Connect (serverless)."""
    try:
        return spark.sparkContext
    except Exception:  # noqa: BLE001
        return None


def run_cancellable(spark: SparkSession, fn: Callable[[], object], timeout_s: int, cancel_grace_s: int = 30):
    """
    Run fn() in a thread. If it overruns, cancel its Spark work.

    Spark Connect (serverless, Databricks Connect): cancelled by tag via interruptTag.
    Classic sessions: interruptTag matches nothing there (verified locally), so the
    worker also sets a job group, and the timeout path cancels that group instead.
    """
    tag = f"ai-guard-{uuid.uuid4()}"
    sc = _classic_context(spark)
    out: dict = {}

    def work():
        # Tag / job group are set inside the worker so they cover the queries fn() launches.
        spark.addTag(tag)
        if sc is not None:
            sc.setJobGroup(tag, "ai_guard chunk", interruptOnCancel=True)
        try:
            out["value"] = fn()
        except BaseException as e:  # noqa: BLE001
            out["error"] = e
        finally:
            try:
                spark.removeTag(tag)
            except Exception:  # noqa: BLE001
                pass

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout_s)

    if t.is_alive():
        log.warning("Timeout after %ss, cancelling work tagged %s", timeout_s, tag)
        interrupted = spark.interruptTag(tag)
        if not interrupted:
            log.warning("interruptTag matched no running operation for %s", tag)
            if sc is not None:
                log.warning("Classic session: cancelling the job group instead")
                sc.cancelJobGroup(tag)
        t.join(cancel_grace_s)
        if t.is_alive():
            # Cancel did not take. Do NOT start another chunk on top of it.
            raise RunAborted(
                f"Interrupt sent but the query was still running {cancel_grace_s}s later. "
                "Check query history in the workspace and cancel it manually."
            )
        raise ChunkTimeout(f"Chunk exceeded {timeout_s}s and was cancelled")

    if "error" in out:
        raise out["error"]
    return out.get("value")


def _pending_df(spark: SparkSession, source_df: DataFrame, target_table: str, id_col: str) -> DataFrame:
    """Ids still to do."""
    ids = source_df.select(id_col).distinct()
    if spark.catalog.tableExists(target_table):
        done = spark.table(target_table).where(F.col("error_message").isNull()).select(id_col)
        ids = ids.join(done, on=id_col, how="left_anti")
    return ids


def _check_contract(out_df: DataFrame, id_col: str) -> None:
    missing = {id_col, "error_message"} - set(out_df.columns)
    if missing:
        raise ValueError(f"transform output is missing required column(s): {sorted(missing)}")


def _check_unique_ids(source_df: DataFrame, id_col: str) -> None:
    """Ids must be unique and non-null, or duplicate rows would be sent (and billed) twice."""
    r = source_df.agg(
        F.count(F.lit(1)).alias("n"),
        F.count(id_col).alias("n_nonnull"),
        F.countDistinct(id_col).alias("n_distinct"),
    ).collect()[0]
    extra = r["n"] - r["n_distinct"]
    if extra > 0:
        raise ValueError(
            f"id column '{id_col}' must be unique and non-null: {extra} extra row(s) come from "
            f"duplicate ids ({r['n'] - r['n_nonnull']} null). No model calls were made."
        )


def _validate_limit_config(source_df: DataFrame, cfg: GuardConfig) -> None:
    if not (cfg.max_input_chars or cfg.max_total_input_chars):
        return
    if not cfg.input_col:
        which = "max_input_chars" if cfg.max_input_chars else "max_total_input_chars"
        raise ValueError(f"{which} is set but input_col is not")
    if cfg.input_col not in source_df.columns:
        raise ValueError(f"input_col '{cfg.input_col}' is not a column of source_df")
    if cfg.max_input_chars and cfg.oversize_policy not in ("truncate", "skip", "fail"):
        raise ValueError(f"unknown oversize_policy: {cfg.oversize_policy}")


def _preflight_input_scan(source_df: DataFrame, cfg: GuardConfig) -> None:
    """Cheap scan of the source (no model calls): report text lengths, enforce 'fail'."""
    length = F.length(F.col(cfg.input_col))
    r = source_df.agg(
        F.count("*").alias("n"),
        F.max(length).alias("max_len"),
        F.percentile_approx(length, 0.99).alias("p99_len"),
        F.sum((length > cfg.max_input_chars).cast("int")).alias("n_over"),
    ).collect()[0]
    log.info(
        "Input scan on '%s': %s rows, max %s chars, p99 %s chars, %s rows over the %s limit (policy=%s)",
        cfg.input_col, r["n"], r["max_len"], r["p99_len"], r["n_over"], cfg.max_input_chars, cfg.oversize_policy,
    )
    if cfg.oversize_policy == "fail" and (r["n_over"] or 0) > 0:
        raise ValueError(
            f"{r['n_over']} rows exceed max_input_chars={cfg.max_input_chars} and oversize_policy='fail'. "
            "No model calls were made."
        )


def _check_total_input(source_df: DataFrame, pending_df: DataFrame, cfg: GuardConfig, id_col: str) -> int:
    """Estimate the characters that will be sent for pending rows; refuse if over the cap."""
    length = F.length(F.col(cfg.input_col))
    if cfg.max_input_chars:
        if cfg.oversize_policy == "truncate":
            sent = F.least(length, F.lit(cfg.max_input_chars))
        else:  # skip (and fail): oversized rows are not sent
            sent = F.when(length <= cfg.max_input_chars, length).otherwise(F.lit(0))
    else:
        sent = length
    total = (
        source_df.join(pending_df, on=id_col, how="inner")
        .agg(F.coalesce(F.sum(sent), F.lit(0)).alias("t"))
        .collect()[0]["t"]
    )
    total = int(total or 0)
    log.info("Estimated input volume for this run: %d characters (cap %s)", total, cfg.max_total_input_chars)
    if cfg.max_total_input_chars and total > cfg.max_total_input_chars:
        raise ValueError(
            f"Estimated {total} input characters for this run, over max_total_input_chars="
            f"{cfg.max_total_input_chars}. No model calls were made."
        )
    return total


def _build_output(chunk_df: DataFrame, transform: Transform, cfg: GuardConfig, id_col: str) -> DataFrame:
    """Apply the input limit, run the transform, and return the chunk's result rows."""
    if not cfg.max_input_chars:
        return transform(chunk_df)

    col = F.col(cfg.input_col)
    too_long = F.length(col) > cfg.max_input_chars

    if cfg.oversize_policy == "truncate":
        flags = chunk_df.select(id_col, too_long.alias("input_truncated"))
        limited = chunk_df.withColumn(cfg.input_col, F.substring(col, 1, cfg.max_input_chars))
        return (
            transform(limited)
            .join(flags, on=id_col, how="left")
            .withColumn("skip_reason", F.lit(None).cast("string"))
        )

    # "skip", and also the safety net for "fail": oversized rows never reach the model.
    ok_df = chunk_df.where(~too_long | col.isNull())
    skipped = (
        chunk_df.where(too_long)
        .select(id_col)
        .withColumn("error_message", F.lit(None).cast("string"))
        .withColumn("skip_reason", F.lit("input_too_long"))
    )
    scored = (
        transform(ok_df)
        .withColumn("skip_reason", F.lit(None).cast("string"))
        .withColumn("input_truncated", F.lit(False))
    )
    skipped = skipped.withColumn("input_truncated", F.lit(False))
    return scored.unionByName(skipped, allowMissingColumns=True)


def _send_alert(cfg: GuardConfig, exc: BaseException, target_table: str, progress: dict) -> None:
    event = {
        "event": "ai_guard_aborted",
        "kind": type(exc).__name__,
        "error": str(exc),
        "target_table": target_table,
        "at": datetime.now(timezone.utc).isoformat(),
        **progress,
    }
    log.error("ai_guard aborted: %s", event)
    if cfg.on_alert is None:
        return
    try:
        cfg.on_alert(event)
    except Exception:  # noqa: BLE001
        log.exception("on_alert hook failed; ignoring so the original error is not masked")


def run_guarded(
    spark: SparkSession,
    source_df: DataFrame,
    target_table: str,
    transform: Transform,
    cfg: GuardConfig = GuardConfig(),
    id_col: str = "id",
) -> dict:
    run_id = uuid.uuid4().hex[:8]
    progress = {"run_id": run_id, "rows_written": 0, "rows_with_errors": 0,
                "chunks_done": 0, "chunks_failed": 0, "chunks_total": None}
    try:
        return _run(spark, source_df, target_table, transform, cfg, id_col, run_id, progress)
    except Exception as e:  # noqa: BLE001
        _send_alert(cfg, e, target_table, progress)
        raise


def _run(spark, source_df, target_table, transform, cfg, id_col, run_id, progress) -> dict:
    t0 = time.monotonic()

    def remaining() -> float | None:
        return None if cfg.max_run_seconds is None else cfg.max_run_seconds - (time.monotonic() - t0)

    def check_deadline() -> None:
        r = remaining()
        if r is not None and r <= 0:
            raise RunAborted(
                f"Run deadline of {cfg.max_run_seconds}s reached. "
                f"{progress['rows_written']} rows written so far; rerun to resume."
            )

    _validate_limit_config(source_df, cfg)
    _check_unique_ids(source_df, id_col)

    if cfg.statement_timeout_s:
        try:
            spark.sql(f"SET STATEMENT_TIMEOUT = {int(cfg.statement_timeout_s)}")
        except Exception as e:  # noqa: BLE001
            log.warning("Could not set STATEMENT_TIMEOUT (%s). The watchdog is the only guard.", e)

    if cfg.max_input_chars:
        _preflight_input_scan(source_df, cfg)

    pending_df = _pending_df(spark, source_df, target_table, id_col)
    pending = [row[0] for row in pending_df.collect()]
    n_pending = len(pending)
    log.info("Run %s: %d rows pending", run_id, n_pending)

    if n_pending > cfg.max_rows:
        raise ValueError(
            f"{n_pending} rows pending, over max_rows={cfg.max_rows}. "
            "Raise the cap deliberately or narrow the source."
        )

    est_chars = None
    if cfg.input_col and (cfg.max_total_input_chars or cfg.max_input_chars):
        est_chars = _check_total_input(source_df, pending_df, cfg, id_col)

    chunks = [pending[i : i + cfg.chunk_size] for i in range(0, n_pending, cfg.chunk_size)]
    progress["chunks_total"] = len(chunks)
    id_schema = source_df.select(id_col).schema  # keeps the id's real Spark type
    rows_written = 0
    rows_with_errors = 0
    rows_skipped = 0
    rows_truncated = 0
    consecutive_failures = 0
    failed_chunks = 0
    contract_checked = False

    for i, chunk_ids in enumerate(chunks):
        check_deadline()
        batch_id = f"{run_id}-{i}"
        ids_df = spark.createDataFrame([(x,) for x in chunk_ids], schema=id_schema)
        chunk_df = source_df.join(ids_df, on=id_col, how="inner")

        out_df = (
            _build_output(chunk_df, transform, cfg, id_col)
            .withColumn("processed_at", F.current_timestamp())
            .withColumn("batch_id", F.lit(batch_id))
        )
        if not contract_checked:
            _check_contract(out_df, id_col)  # schema check only, no compute
            contract_checked = True

        def run_chunk(df=out_df):
            # Compute and write both happen inside the cancellable thread.
            # A Delta append is atomic, so a cancel before commit writes nothing.
            # mergeSchema lets a target created before the input limit was turned
            # on pick up the input_truncated / skip_reason columns.
            df.write.mode("append").option("mergeSchema", "true").saveAsTable(target_table)

        ok = False
        t_chunk = time.monotonic()
        for attempt in range(cfg.max_retries + 1):
            check_deadline()
            r = remaining()
            chunk_timeout = cfg.timeout_s if r is None else max(1, min(cfg.timeout_s, int(r)))
            try:
                run_cancellable(spark, run_chunk, chunk_timeout, cfg.cancel_grace_s)
                ok = True
                break
            except RunAborted:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("Chunk %d attempt %d failed: %r", i, attempt + 1, e)
                if attempt < cfg.max_retries:
                    time.sleep(cfg.retry_backoff_s * (attempt + 1))

        if not ok:
            consecutive_failures += 1
            failed_chunks += 1
            progress["chunks_failed"] = failed_chunks
            if consecutive_failures >= cfg.max_consecutive_failures:
                raise RunAborted(
                    f"{consecutive_failures} chunks in a row failed. Stopping. "
                    f"{rows_written} rows written so far; rerun to resume."
                )
            continue

        dur = time.monotonic() - t_chunk
        if dur > cfg.slow_chunk_fraction * cfg.timeout_s:
            log.warning(
                "Chunk %d took %.0fs, over %.0f%% of the %ss timeout. The service may be degrading.",
                i, dur, cfg.slow_chunk_fraction * 100, cfg.timeout_s,
            )

        consecutive_failures = 0
        aggs = [F.count("*").alias("n"), F.count("error_message").alias("e")]
        if cfg.max_input_chars:
            aggs += [
                F.count("skip_reason").alias("skipped"),
                F.sum(F.col("input_truncated").cast("int")).alias("truncated"),
            ]
        stats = spark.table(target_table).where(F.col("batch_id") == batch_id).agg(*aggs).collect()[0]
        rows_written += stats["n"]
        rows_with_errors += stats["e"]
        if cfg.max_input_chars:
            rows_skipped += stats["skipped"] or 0
            rows_truncated += stats["truncated"] or 0
        progress.update(rows_written=rows_written, rows_with_errors=rows_with_errors, chunks_done=i + 1)
        log.info("Chunk %d/%d ok: %d rows, %d with errors", i + 1, len(chunks), stats["n"], stats["e"])

        # Skipped rows have a null error_message on purpose, so they do not trip this.
        if stats["n"] and stats["e"] / stats["n"] > cfg.max_error_rate:
            raise RunAborted(
                f"Error rate {stats['e']}/{stats['n']} in chunk {i} is over {cfg.max_error_rate:.0%}. "
                "Stopping so a sick service does not keep burning money."
            )

    # A run with failed chunks is NOT a success, even if no breaker tripped: those rows were
    # not processed. Raising here makes the job fail and the alert fire instead of reporting
    # success with work silently left undone. Rerunning resumes with just those rows.
    if failed_chunks:
        raise RunAborted(
            f"{failed_chunks} of {len(chunks)} chunk(s) failed and were skipped. "
            f"{rows_written} of {n_pending} pending rows were processed; the rest were NOT. "
            "Rerun to resume."
        )

    return {
        "run_id": run_id,
        "pending_at_start": n_pending,
        "rows_written": rows_written,
        "rows_with_errors": rows_with_errors,
        "rows_skipped_oversize": rows_skipped,
        "rows_truncated": rows_truncated,
        "estimated_input_chars": est_chars,
        "chunks": len(chunks),
        "elapsed_s": round(time.monotonic() - t0, 1),
    }
