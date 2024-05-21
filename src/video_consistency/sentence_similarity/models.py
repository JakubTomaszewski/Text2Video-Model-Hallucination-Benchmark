from sentence_transformers import SentenceTransformer, util


class SentenceSimilarityCalculator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str="cuda", **kwargs):
        self.device = device
        self.model = SentenceTransformer(model_name).to(device)

    def evaluate(self, sentence_1: str, sentence_2: str) -> float:
        """Calculate the similarity between two sentences.

        Args:
            sentence_1 (str): sentence 1
            sentence_2 (str): sentence 2

        Returns:
            float: similarity between the two sentences
        """
        return self._calculate_similarity(sentence_1, sentence_2)

    def _calculate_similarity(self, sentence_1: str, sentence_2: str) -> float:
        """Calculate the cosine similarity between two sentences.

        Args:
            sentence_1 (str): sentence 1
            sentence_2 (str): sentence 2

        Returns:
            float: cosine similarity between the two sentences
        """
        emb1 = self.model.encode(sentence_1)
        emb2 = self.model.encode(sentence_2)
        return util.cos_sim(emb1, emb2)
