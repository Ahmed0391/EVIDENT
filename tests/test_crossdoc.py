from evident.pipeline import crossdoc


def _doc(doc_id, **fields):
    return {"doc_id": doc_id, "fields": fields}


def test_matching_names():
    docs = [_doc("id", full_name="Ahmed Saidi"), _doc("poa", full_name="Ahmed Saidi")]
    assert crossdoc.check_names(docs).status == "match"


def test_name_conflict():
    docs = [_doc("id", full_name="Ahmed Saidi"), _doc("poa", full_name="Bilel Trabelsi")]
    assert crossdoc.check_names(docs).status == "conflict"


def test_minor_typo_still_matches():
    # transliteration drift should stay above threshold
    docs = [_doc("id", full_name="Mohamed Saidi"), _doc("pp", full_name="Mohammed Saidi")]
    assert crossdoc.check_names(docs).status == "match"


def test_single_source_name():
    docs = [_doc("poa", full_name="Ahmed Saidi")]
    assert crossdoc.check_names(docs).status == "single_source"


def test_conflicting_dob():
    docs = [_doc("id", dob="30/10/2003"), _doc("pp", dob="30/10/2001")]
    assert crossdoc.check_dob(docs).status == "conflict"
