import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from classifier_config import AIClassifierConfig

LABELS = {"liability": "who is at fault", "other": "anything else"}
ALWAYS = {"version": "2.1", "enableConfidenceScores": "true", "enableRationales": "true"}


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("classifier-config-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture()
def fake_ai_classify(monkeypatch):
    """Stand-in for the Databricks-only F.ai_classify, returning the 2.1 output shape.

    Records (labels, options) for every call. Text containing "urgent" gets the label
    "urgent", text containing "FAIL" comes back as an error, anything else gets the
    first label.
    """
    calls = []

    def fake(col, labels, options=None):
        calls.append((labels, options))
        text = F.coalesce(col.cast("string"), F.lit(""))
        label = F.when(text.contains("urgent"), F.lit("urgent")).otherwise(F.lit(next(iter(labels))))
        failed = text.contains("FAIL")
        item = F.concat(F.lit('[{"value":"'), label,
                        F.lit('","confidence_score":0.9,"rationale":"because"}]'))
        return F.parse_json(F.concat(
            F.lit('{"response":'), F.when(failed, F.lit("[]")).otherwise(item),
            F.lit(',"error_message":'),
            F.when(failed, F.lit('"model failed"')).otherwise(F.lit("null")),
            F.lit("}"),
        ))

    monkeypatch.setattr(F, "ai_classify", fake, raising=False)
    return calls


def make_src(spark, rows):
    return spark.createDataFrame(rows, "id long, text string")


def run(spark, cfg, rows):
    return {r["id"]: r for r in cfg.transform("text", "id")(make_src(spark, rows)).collect()}


# ---------------------------------------------------------------- options

def test_options_pin_version_and_always_ask_for_confidence_and_rationale():
    assert AIClassifierConfig(LABELS).options() == ALWAYS


def test_options_include_instructions_when_given():
    opts = AIClassifierConfig(LABELS, instructions="claims notes").options()
    assert opts == {**ALWAYS, "instructions": "claims notes"}


def test_options_leave_out_empty_instructions():
    assert "instructions" not in AIClassifierConfig(LABELS, instructions="").options()


# -------------------------------------------------------------- transform

def test_output_columns_and_types(spark, fake_ai_classify):
    df = AIClassifierConfig(LABELS).transform("text", "id")(make_src(spark, [(1, "a")]))
    assert df.columns == ["id", "label", "confidence", "rationale", "error_message",
                          "raw_response", "config_version", "function_version"]
    assert dict(df.dtypes)["confidence"] == "double"


def test_values_are_unwrapped(spark, fake_ai_classify):
    out = run(spark, AIClassifierConfig(LABELS), [(1, "this is urgent"), (2, "meh")])
    assert (out[1]["label"], out[1]["confidence"], out[1]["rationale"]) == ("urgent", 0.9, "because")
    assert out[2]["label"] == "liability"
    assert out[1]["error_message"] is None
    assert '"value":"urgent"' in out[1]["raw_response"].replace(" ", "")


def test_errors_surface_in_error_message_with_empty_fields(spark, fake_ai_classify):
    row = run(spark, AIClassifierConfig(LABELS), [(1, "FAIL me")])[1]
    assert row["error_message"] == "model failed"
    assert row["label"] is None and row["confidence"] is None and row["rationale"] is None


def test_versions_are_stamped_on_every_row(spark, fake_ai_classify):
    out = run(spark, AIClassifierConfig(LABELS, config_version="7"), [(1, "a"), (2, "b")])
    assert {(r["config_version"], r["function_version"]) for r in out.values()} == {("7", "2.1")}


def test_labels_and_options_reach_ai_classify(spark, fake_ai_classify):
    run(spark, AIClassifierConfig(LABELS, instructions="notes"), [(1, "a")])
    assert fake_ai_classify[-1] == (LABELS, {**ALWAYS, "instructions": "notes"})


def test_one_output_row_per_input_row_and_null_text_is_handled(spark, fake_ai_classify):
    out = run(spark, AIClassifierConfig(LABELS), [(1, "a"), (2, None), (3, "c")])
    assert sorted(out) == [1, 2, 3]


def test_id_col_and_text_col_names_are_used(spark, fake_ai_classify):
    df = spark.createDataFrame([("C-1", "this is urgent")], "claim_number string, note_text string")
    out = AIClassifierConfig(LABELS).transform("note_text", "claim_number")(df).collect()
    assert out[0]["claim_number"] == "C-1" and out[0]["label"] == "urgent"


def test_other_input_columns_are_not_carried_through(spark, fake_ai_classify):
    df = spark.createDataFrame([(1, "a", "keep?")], "id long, text string, extra string")
    assert "extra" not in AIClassifierConfig(LABELS).transform("text", "id")(df).columns


# ------------------------------------------------- how it is used (join)

def test_result_joins_back_to_the_claims_on_the_id(spark, fake_ai_classify):
    claims = spark.createDataFrame(
        [(1, "this is urgent", "A"), (2, "meh", "B")], "id long, text string, state string")
    result = AIClassifierConfig(LABELS).transform("text", "id")(claims)
    joined = {r["id"]: r for r in claims.join(result, on="id", how="left").collect()}
    assert joined[1]["state"] == "A" and joined[1]["label"] == "urgent"
    assert joined[2]["state"] == "B" and joined[2]["label"] == "liability"
