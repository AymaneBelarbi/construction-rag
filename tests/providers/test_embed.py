import math
import unittest

from rag.providers.embed import HASHING_ID, hashing_embed, min_vector_score_for, tokenize, tokenize_query


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))  # vectors are unit length, so dot product = cosine


class EmbedTests(unittest.TestCase):
    def test_tokenize_keeps_ids_whole_and_in_parts(self) -> None:
        tokens = tokenize("RFI-003 is open")
        for expected in ("rfi-003", "rfi", "003", "open"):
            self.assertIn(expected, tokens)

    def test_letters_with_accents_are_kept_whole(self) -> None:
        self.assertEqual(tokenize("Müller Électrique, café"), ["müller", "électrique", "café"])

    def test_a_lone_letter_is_not_a_word(self) -> None:
        # "contractor's" must not leave a stray 's' that matches every S-xxx record.
        self.assertEqual(tokenize("contractor's note"), ["contractor", "note"])
        self.assertIn("s-001", tokenize("S-001"))

    def test_a_query_keeps_an_id_as_one_exact_token_and_does_not_split_it(self) -> None:
        self.assertEqual(tokenize_query("What is RFI-003?"), ["rfi-003"])  # 'what' and 'is' are stop words
        self.assertEqual(tokenize_query("DR-2026-08-03"), ["dr-2026-08-03"])

    def test_stop_words_are_dropped_from_a_question_but_kept_in_the_text_being_indexed(self) -> None:
        self.assertEqual(tokenize_query("Which RFIs are overdue for the architect?"), ["rfis", "overdue", "architect"])
        self.assertIn("for", tokenize("waiting for the architect"))  # the indexed text keeps every word

    def test_the_hashing_embedder_id_says_which_tokenizer_made_the_vectors(self) -> None:
        self.assertIn("v3", HASHING_ID)  # bumped when the tokenizer changed, so old cached vectors are rebuilt

    def test_common_words_do_not_make_two_texts_look_similar(self) -> None:
        # "recipe for chocolate cake" must not resemble a record just because both contain "for".
        recipe, record = hashing_embed(["recipe for chocolate cake", "Casework finish needed for millwork shop drawings"])
        self.assertLess(cosine(recipe, record), 0.15)

    def test_the_similarity_cut_off_depends_on_the_embedding_model(self) -> None:
        self.assertEqual(min_vector_score_for(HASHING_ID, {}), 0.32)
        small = min_vector_score_for("api:https://api.openai.com/v1:text-embedding-3-small", {})
        self.assertGreater(small, 0.2)  # real embeddings put unrelated texts higher than the hashing embedder does
        self.assertEqual(min_vector_score_for("api:x:some-future-model", {}), 0.2)  # unknown model: the safe default

    def test_the_cut_off_can_be_set_in_the_environment_and_a_bad_value_is_an_error(self) -> None:
        self.assertEqual(min_vector_score_for(HASHING_ID, {"RAG_MIN_VECTOR_SCORE": "0.35"}), 0.35)
        for bad in ("high", "-0.1", "1.5", "nan"):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    min_vector_score_for(HASHING_ID, {"RAG_MIN_VECTOR_SCORE": bad})

    def test_vectors_are_unit_length_and_same_size(self) -> None:
        vectors = hashing_embed(["open rfi about the beam", "paint colours"])
        self.assertEqual(len(vectors[0]), len(vectors[1]))
        for vector in vectors:
            self.assertAlmostEqual(math.sqrt(sum(x * x for x in vector)), 1.0, places=6)

    def test_same_text_gives_same_vector(self) -> None:
        self.assertEqual(hashing_embed(["beam size"]), hashing_embed(["beam size"]))

    def test_similar_text_is_closer_than_unrelated_text(self) -> None:
        base, similar, other = hashing_embed(
            ["open rfi about beam size", "beam size rfi open", "interior paint colours sample"]
        )
        self.assertGreater(cosine(base, similar), cosine(base, other))

    def test_empty_text_does_not_crash(self) -> None:
        (vector,) = hashing_embed([""])
        self.assertTrue(all(x == 0.0 for x in vector))


if __name__ == "__main__":
    unittest.main()
