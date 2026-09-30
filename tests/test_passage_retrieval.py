from phase5_helpers import OfflineTest, page, paper, passage
from researchpilot.passage_retrieval import LexicalRetriever, PageChunker


class PassageTests(OfflineTest):
    def test_chunks_do_not_cross_pages_or_papers(self):
        pages = [page(number=1), page(number=3), page(paper("W2"), None)]
        chunks = PageChunker().chunk(pages)
        self.assertEqual([(p.paper_id, p.page_number) for p in chunks],
                         [("openalex:W1", 1), ("openalex:W1", 3), ("openalex:W2", None)])
        self.assertEqual(chunks[-1].source_type, "abstract")

    def test_chunk_ids_are_deterministic_and_content_dependent(self):
        chunker = PageChunker()
        self.assertEqual(chunker.chunk([page()]), chunker.chunk([page()]))
        self.assertNotEqual(chunker.chunk([page()])[0].passage_id,
                            chunker.chunk([page(text="Other evidence on storage capacity and latency.")])[0].passage_id)

    def test_long_page_is_split_with_overlap_and_no_tiny_tail(self):
        text = " ".join(f"word{i}" for i in range(400))
        chunks = PageChunker(400, 60).chunk([page(text=text)])
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 480)
            self.assertGreater(len(chunk.text), 30)
            self.assertEqual(chunk.page_number, 1)
        self.assertTrue(set(chunks[0].text.split()) & set(chunks[1].text.split()))
        self.assertEqual(set(text.split()), set(" ".join(c.text for c in chunks).split()))

    def test_tiny_headers_are_skipped(self):
        self.assertEqual(PageChunker().chunk([page(text="Page 1")] ), [])

    def test_invalid_chunk_configuration(self):
        for size, overlap in [(True, 0), (100, 0), (400, 300), (400, -1)]:
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                PageChunker(size, overlap)

    def test_lexical_relevance_top_k_and_zero_matches(self):
        items = [passage("a", text="Shared caches reduce storage latency for repeated reads."),
                 passage("b", text="Ocean currents influence sea temperatures and weather."),
                 passage("c", text="Storage capacity limits cache allocation for workloads.")]
        result = LexicalRetriever().retrieve("shared caches storage latency", items, 1)
        self.assertEqual(result[0].passage.passage_id, "a")
        self.assertGreater(result[0].lexical_score, 0)
        self.assertNotIn("b", [r.passage.passage_id for r in LexicalRetriever().retrieve("storage", items)])

    def test_lexical_ties_preserve_input_order(self):
        result = LexicalRetriever().retrieve("storage", [passage("b"), passage("a")])
        self.assertEqual([r.passage.passage_id for r in result], ["b", "a"])

    def test_empty_query_documents_or_no_meaningful_match(self):
        for query, passages in [("storage", []), ("", [passage()]), ("how does the", [passage()]), ("ocean", [passage()])]:
            self.assertEqual(LexicalRetriever().retrieve(query, passages), [])

    def test_lexical_unicode_case_normalization(self):
        self.assertEqual(len(LexicalRetriever().retrieve("ＳＴＯＲＡＧＥ", [passage()])), 1)

    def test_duplicate_passage_ids_rejected(self):
        with self.assertRaises(ValueError):
            LexicalRetriever().retrieve("storage", [passage(), passage()])
