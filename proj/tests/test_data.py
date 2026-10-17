import pandas as pd

from cleantext.data import digest, norm_text, remove_cross_split_duplicates


def _df(texts):
    return pd.DataFrame({"text": texts, "label": ["a"] * len(texts)})


def test_normalisation_ignores_case_and_whitespace():
    assert norm_text("  Привіт   СВІТ \n") == "привіт світ"
    assert digest("Привіт  світ") == digest(" привіт світ ")
    assert digest("а") != digest("б")


def test_cross_split_priority_test_val_train():
    train = _df(["Один", "Два", "Три", "Чотири"])
    val = _df(["два", "Пять"])
    test = _df(["ТРИ  ", "Шість"])
    tr, va, te, removed = remove_cross_split_duplicates(train, val, test)
    assert te.text.tolist() == ["ТРИ  ", "Шість"]      # test має найвищий пріоритет
    assert va.text.tolist() == ["два", "Пять"]          # val зберігається, якщо немає в test
    assert tr.text.tolist() == ["Один", "Чотири"]       # «Два» (є у val) і «Три» (є в test) вилучено з train
    assert removed == {"test": 0, "val": 0, "train": 2}


def test_val_loses_to_test():
    _, va, te, removed = remove_cross_split_duplicates(_df(["x"]), _df(["Той самий", "інший"]), _df(["той  самий"]))
    assert va.text.tolist() == ["інший"] and removed["val"] == 1 and len(te) == 1


def test_duplicates_inside_split_removed():
    tr, _, _, removed = remove_cross_split_duplicates(_df(["a b", "A  B", "c"]), _df(["x"]), _df(["y"]))
    assert len(tr) == 2 and removed["train"] == 1
