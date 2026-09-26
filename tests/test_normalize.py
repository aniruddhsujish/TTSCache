from tts_cache.normalize import normalize_basic, normalize


def test_collapses_extra_whitespace():
    assert normalize_basic("Your  order  has\tshipped  .") == "Your order has shipped ."


def test_keeps_case_and_punc():
    assert normalize_basic("Your order has shipped.") != normalize_basic(
        "Your order has shipped!"
    )
    assert normalize_basic("Contact US support") == "Contact US support"


def test_keeps_zero_width_joiner():
    assert normalize_basic("\u200db").startswith("\u200d")


def test_nfc_makes_nukta_forms_identical():
    single = "\u0958"  # क़ stored as one character
    combined = "\u0915\u093c"  # क + nukta dot, stored as two characters

    assert single != combined  # proves the inputs really differ as stored
    assert normalize_basic(single) == normalize_basic(combined)


def test_hindi_currency_variants_become_identical():
    a = normalize("₹500", "hi")
    b = normalize("Rs. 500", "hi")
    c = normalize("रु 500", "hi")

    assert a == b == c
    assert a == "500 रुपये"


def test_canonical_form_is_unchanged():
    assert normalize("500 रुपये", "hi") == "500 रुपये"


def test_no_rule_file_for_lang():
    assert normalize("Rs.  500", "ta") == "Rs. 500"


def test_layer_one_still_runs_for_hindi():
    assert normalize("Rs.   500", "hi") == "500 रुपये"


def test_sentence_ending_dot_is_not_swallowed():
    assert normalize("Pay Rs. 500.", "hi") == "Pay 500 रुपये."
