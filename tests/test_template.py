from tts_cache.strategies.template import find_variables, make_template


def test_finds_one_variable_in_the_middle():
    words = "Your order 4521 has been shipped.".split()
    assert find_variables(words) == [2]


def test_find_one_variable_in_the_end():
    words = "Your ticket number is 4521.".split()
    assert find_variables(words) == [4]


def test_find_two_variables():
    words = "Your order 4521 will arrive in 3 days.".split()
    assert find_variables(words) == [2, 6]


def test_rejects_three_variables():
    words = "Order 1 of 2 costs 45,230.50 rupees.".split()
    assert find_variables(words) is None


def test_no_variables():
    words = "Thank you for calling.".split()
    assert find_variables(words) is None


def test_find_one_variable_hindi():
    words = "आपका ऑर्डर 4521 भेज दिया गया है।".split()
    assert find_variables(words) == [2]


def test_converts_to_template_with_single_variable():
    words = "Your ticket number is 4521.".split()
    assert make_template(words, [4]) == (
        "Your ticket number is {NUM}.",
        ["4521."],
    )


def test_converts_to_template_double_variable():
    words = "Your order 4521 will arrive in 3 days.".split()
    assert make_template(words, [2, 6]) == (
        "Your order {NUM} will arrive in {NUM} days.",
        ["4521", "3"],
    )
