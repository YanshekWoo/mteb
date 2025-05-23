from __future__ import annotations

import json
import logging
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer

from mteb import MTEB, ModelMeta

logging.basicConfig(level=logging.INFO)


def test_mteb_rerank(tmp_path: Path):
    # Test that reranking works
    # unfortunately, we need all the query ids to pretend to have this
    scifact_keys = [
        "1",
        "3",
        "5",
        "13",
        "36",
        "42",
        "48",
        "49",
        "50",
        "51",
        "53",
        "54",
        "56",
        "57",
        "70",
        "72",
        "75",
        "94",
        "99",
        "100",
        "113",
        "115",
        "118",
        "124",
        "127",
        "128",
        "129",
        "130",
        "132",
        "133",
        "137",
        "141",
        "142",
        "143",
        "146",
        "148",
        "163",
        "171",
        "179",
        "180",
        "183",
        "185",
        "198",
        "208",
        "212",
        "213",
        "216",
        "217",
        "218",
        "219",
        "230",
        "232",
        "233",
        "236",
        "237",
        "238",
        "239",
        "248",
        "249",
        "261",
        "268",
        "269",
        "274",
        "275",
        "279",
        "294",
        "295",
        "298",
        "300",
        "303",
        "312",
        "314",
        "324",
        "327",
        "338",
        "343",
        "350",
        "354",
        "362",
        "380",
        "384",
        "385",
        "386",
        "388",
        "399",
        "410",
        "411",
        "415",
        "421",
        "431",
        "436",
        "437",
        "439",
        "440",
        "443",
        "452",
        "475",
        "478",
        "491",
        "501",
        "502",
        "507",
        "508",
        "513",
        "514",
        "516",
        "517",
        "521",
        "525",
        "527",
        "528",
        "532",
        "533",
        "535",
        "536",
        "539",
        "540",
        "544",
        "549",
        "551",
        "552",
        "554",
        "560",
        "569",
        "575",
        "577",
        "578",
        "587",
        "589",
        "593",
        "597",
        "598",
        "613",
        "619",
        "623",
        "628",
        "636",
        "637",
        "641",
        "644",
        "649",
        "659",
        "660",
        "674",
        "684",
        "690",
        "691",
        "692",
        "693",
        "700",
        "702",
        "715",
        "716",
        "718",
        "721",
        "723",
        "727",
        "728",
        "729",
        "742",
        "743",
        "744",
        "756",
        "759",
        "768",
        "770",
        "775",
        "781",
        "783",
        "784",
        "785",
        "793",
        "800",
        "805",
        "808",
        "811",
        "814",
        "820",
        "821",
        "823",
        "830",
        "831",
        "832",
        "834",
        "837",
        "839",
        "845",
        "847",
        "852",
        "859",
        "870",
        "873",
        "879",
        "880",
        "882",
        "887",
        "903",
        "904",
        "907",
        "911",
        "913",
        "914",
        "921",
        "922",
        "936",
        "956",
        "957",
        "960",
        "967",
        "971",
        "975",
        "982",
        "985",
        "993",
        "1012",
        "1014",
        "1019",
        "1020",
        "1021",
        "1024",
        "1029",
        "1041",
        "1049",
        "1062",
        "1086",
        "1088",
        "1089",
        "1099",
        "1100",
        "1104",
        "1107",
        "1110",
        "1121",
        "1130",
        "1132",
        "1137",
        "1140",
        "1144",
        "1146",
        "1150",
        "1163",
        "1175",
        "1179",
        "1180",
        "1185",
        "1187",
        "1191",
        "1194",
        "1196",
        "1197",
        "1199",
        "1200",
        "1202",
        "1204",
        "1207",
        "1213",
        "1216",
        "1221",
        "1225",
        "1226",
        "1232",
        "1241",
        "1245",
        "1259",
        "1262",
        "1266",
        "1270",
        "1271",
        "1272",
        "1273",
        "1274",
        "1278",
        "1279",
        "1280",
        "1281",
        "1282",
        "1290",
        "1292",
        "1298",
        "1303",
        "1316",
        "1319",
        "1320",
        "1332",
        "1335",
        "1336",
        "1337",
        "1339",
        "1344",
        "1352",
        "1359",
        "1362",
        "1363",
        "1368",
        "1370",
        "1379",
        "1382",
        "1385",
        "1389",
        "1395",
    ]
    model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
    eval = MTEB(
        tasks=[
            "SciFact",
        ]
    )
    # create fake first stage results
    tmp_file = tmp_path / "tmp.json"
    with open(tmp_file, "w") as f:
        f.write(
            json.dumps(
                {
                    i: {
                        # just two random documents so we can see it works
                        "4983": 0.1,
                        "18670": 0.9,
                        "19238": 0.01,
                    }
                    for i in scifact_keys
                }
            )
        )

    eval.run(
        model,  # type: ignore
        output_folder="tests/results",
        overwrite_results=True,
        eval_splits=["test"],
        top_k=2,
        previous_results=tmp_file,
        save_predictions=True,
    )
    tmp_file.unlink()

    # read in the results
    with open("tests/results/SciFact_default_predictions.json") as f:
        results = json.load(f)

    # check that only the top two results are re-orderd
    assert "19238" not in results["1"]
    assert "4983" in results["1"]
    assert "18670" in results["1"]


def test_reranker_same_ndcg1():
    de_name = "sentence-transformers/average_word_embeddings_komninos"
    revision = "21eec43590414cb8e3a6f654857abed0483ae36e"
    de = SentenceTransformer(de_name, revision=revision)
    ce = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
    ce_revision = "e9ea2688951463fc2791a2ea2ddfce6762900675"
    ce.mteb_model_meta = ModelMeta(
        name="cross-encoder/ms-marco-TinyBERT-L-2-v2",
        languages=["eng-Latn"],
        open_weights=True,
        revision=ce_revision,
        release_date="2021-04-15",
        n_parameters=None,
        memory_usage_mb=None,
        max_tokens=None,
        embed_dim=None,
        license=None,
        public_training_code=None,
        public_training_data=None,
        reference=None,
        similarity_fn_name=None,
        use_instructions=None,
        training_datasets=None,
        framework=["Sentence Transformers", "PyTorch"],
    )

    eval = MTEB(tasks=["SciFact"])
    eval.run(
        de,
        output_folder="tests/results/stage1",
        overwrite_results=True,
        save_predictions=True,
        eval_splits=["test"],
    )
    eval.run(
        ce,  # type: ignore
        output_folder="tests/results/stage2",
        overwrite_results=True,
        previous_results="tests/results/stage1/SciFact_default_predictions.json",
        save_predictions=False,
        eval_splits=["test"],
        top_k=1,  # don't allow it to rerank more than 1 so we can check for top_1 being the same
    )

    # read in stage 1 and stage two and check ndcg@1 is the same
    with open(
        f"tests/results/stage1/{de_name.replace('/', '__')}/{revision}/SciFact.json"
    ) as f:
        stage1 = json.load(f)

    with open(
        f"tests/results/stage2/cross-encoder__ms-marco-TinyBERT-L-2-v2/{ce_revision}/SciFact.json"
    ) as f:
        stage2 = json.load(f)

    assert (
        stage1["scores"]["test"][0]["ndcg_at_1"]
        == stage2["scores"]["test"][0]["ndcg_at_1"]
    )
