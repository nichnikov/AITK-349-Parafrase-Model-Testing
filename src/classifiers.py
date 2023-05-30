"""
классификатор KNeighborsClassifier в /home/an/Data/Yandex.Disk/dev/03-jira-tasks/aitk115-support-questions
"""
from src.data_types import Parameters
from src.storage import ElasticClient
from src.texts_processing import TextsTokenizer
from src.config import logger
from sentence_transformers import util
from collections import namedtuple

# https://stackoverflow.com/questions/492519/timeout-on-a-function-call

tmt = float(10)  # timeout


class FastAnswerClassifier:
    """Объект для оперирования MatricesList и TextsStorage"""

    def __init__(self, tokenizer: TextsTokenizer, parameters: Parameters, model):
        self.es = ElasticClient()
        self.tkz = tokenizer
        self.prm = parameters
        self.model = model

    async def searching(self, text: str, pubid: int, score: float):
        """"""
        """searching etalon by  incoming text"""
        try:
            tokens = self.tkz([text])
            if tokens[0]:
                tokens_str = " ".join(tokens[0])
                etalons_search_result = await self.es.texts_search(self.prm.clusters_index, "LemCluster", [tokens_str])
                result_dicts = etalons_search_result[0]["search_results"]
                if result_dicts:
                    results_tuples = [(d["ID"], d["Cluster"], d["LemCluster"]) for d in result_dicts]
                    text_emb = self.model.encode(text)
                    ids, ets, lm_ets = zip(*results_tuples)
                    candidate_embs = self.model.encode(ets)
                    scores = util.cos_sim(text_emb, candidate_embs)
                    print([score.item() for score in scores[0]])
                    scores_list = [score.item() for score in scores[0]]
                    print([x for x in list(zip(ids, ets, lm_ets, scores_list)) if x[3] > score])
                    the_best_result = sorted(list(zip(ids, ets, lm_ets, scores_list)),
                                             key=lambda x: x[3], reverse=True)[0]
                    if the_best_result[3] >= score:
                        print("Fast Text Score:", the_best_result[3])
                        answers_search_result = await self.es.answer_search(self.prm.answers_index, the_best_result[0],
                                                                            pubid)
                        if answers_search_result["search_results"]:
                            search_result = {"templateId": answers_search_result["search_results"][0]["templateId"],
                                             "templateText": answers_search_result["search_results"][0]["templateText"]}
                            logger.info("search completed successfully with result: {}".format(str(search_result)))
                            return search_result
                        else:
                            logger.info(
                                "not found answer with templateId {} and pub_id {}".format(str(the_best_result[0]),
                                                                                           str(pubid)))
                    else:
                        logger.info("elasticsearch doesn't find any etalons for input text {}".format(str(text)))
                        return {"templateId": 0, "templateText": ""}
                else:
                    logger.info("elasticsearch doesn't find any etalons for input text {}".format(str(text)))
                    return {"templateId": 0, "templateText": ""}
            else:
                logger.info("tokenizer returned empty value for input text {}".format(str(text)))
                return {"templateId": 0, "templateText": ""}
                # return tuple(0, "no", "no", 0)
        except Exception:
            logger.exception("Searching problem with text: {}".format(str(text)))
            # return tuple(0, "no", "no", 0)
            return {"templateId": 0, "templateText": ""}

    async def tested_searching(self, text: str, pubid: int, score: float):
        """"""
        """возвращает несколько скоров и позиций эластика для оценки работы Эластик + Сберт:"""
        SearchResult = namedtuple("SearchResult", "ID, Etalon, LemEtalon, Score")
        try:
            tokens = self.tkz([text])
            if tokens[0]:
                tokens_str = " ".join(tokens[0])
                etalons_search_result = await self.es.texts_search(self.prm.clusters_index, "LemCluster", [tokens_str])
                result_dicts = etalons_search_result[0]["search_results"]
                if result_dicts:
                    results_tuples = [(d["ID"], d["Cluster"], d["LemCluster"]) for d in result_dicts]
                    text_emb = self.model.encode(text)
                    ids, ets, lm_ets = zip(*results_tuples)
                    candidate_embs = self.model.encode(ets)
                    scores = util.cos_sim(text_emb, candidate_embs)
                    scores_list = [score.item() for score in scores[0]]
                    search_result = {}
                    candidates = [SearchResult(*x) for x in list(zip(ids, ets, lm_ets, scores_list)) if x[3]]
                    search_result["ElasticOutputQuantity"] = len(candidates)
                    search_result["FastAnswerQuantity"] = len(set(ids))
                    for i in range(3):
                        try:
                            search_result["ID" + str(i)] = candidates[i].ID
                        except:
                            search_result["ID" + str(i)] = 0
                        try:
                            search_result["Etalon" + str(i)] = candidates[i].Etalon
                        except:
                            search_result["Etalon" + str(i)] = "NO"
                        try:
                            search_result["Score" + str(i)] = candidates[i].Score
                        except:
                            search_result["Score" + str(i)] = 0.0

                    the_best_result = sorted(candidates, key=lambda x: x[3], reverse=True)[0]
                    if the_best_result:
                        search_result["BestID"] = the_best_result.ID
                        search_result["BestEtalon"] = the_best_result.Etalon
                        search_result["BestScore"] = the_best_result.Score
                        answers_search_result = await self.es.answer_search(self.prm.answers_index, the_best_result.ID,
                                                                            pubid)
                        if answers_search_result["search_results"]:
                            search_result["BestAnswer"] = answers_search_result["search_results"][0]["templateText"]
                        search_result["BestElasticNum"] = candidates.index(the_best_result)
                    else:
                        search_result["BestID"] = 0
                        search_result["BestEtalon"] = "NO"
                        search_result["BestScore"] = "NO"
                        search_result["BestAnswer"] = "NO"
                        search_result["BestElasticNum"] = 0
                    return search_result
                else:
                    logger.info("elasticsearch doesn't find any etalons for input text {}".format(str(text)))
                    return {"templateId": 0, "templateText": ""}
            else:
                logger.info("tokenizer returned empty value for input text {}".format(str(text)))
                return {"templateId": 0, "templateText": ""}
                # return tuple(0, "no", "no", 0)
        except Exception:
            logger.exception("Searching problem with text: {}".format(str(text)))
            # return tuple(0, "no", "no", 0)
            return {"templateId": 0, "templateText": ""}

    async def tested_searching_ever_group(self, text: str, pubid: int, score: float):
        """"""
        """возвращает несколько скоров и позиций эластика для оценки работы Эластик + Сберт:"""
        SearchResult = namedtuple("SearchResult", "ID, Etalon, LemEtalon, Score")
        try:
            tokens = self.tkz([text])
            if tokens[0]:
                tokens_str = " ".join(tokens[0])
                etalons_search_result = await self.es.texts_search(self.prm.clusters_index, "LemCluster", [tokens_str])
                result_dicts = etalons_search_result[0]["search_results"]
                if result_dicts:
                    results_tuples = [(d["ID"], d["Cluster"], d["LemCluster"]) for d in result_dicts]
                    text_emb = self.model.encode(text)
                    ids, ets, lm_ets = zip(*results_tuples)
                    candidate_embs = self.model.encode(ets)
                    scores = util.cos_sim(text_emb, candidate_embs)
                    scores_list = [score.item() for score in scores[0]]
                    search_result = {}
                    candidates = [SearchResult(*x) for x in list(zip(ids, ets, lm_ets, scores_list)) if x[3]]
                    search_result["ElasticOutputQuantity"] = len(candidates)
                    search_result["FastAnswerQuantity"] = len(set(ids))
                    for i in range(3):
                        try:
                            search_result["ID" + str(i)] = candidates[i].ID
                        except:
                            search_result["ID" + str(i)] = 0
                        try:
                            search_result["Etalon" + str(i)] = candidates[i].Etalon
                        except:
                            search_result["Etalon" + str(i)] = "NO"
                        try:
                            search_result["Score" + str(i)] = candidates[i].Score
                        except:
                            search_result["Score" + str(i)] = 0.0

                    the_best_result = sorted(candidates, key=lambda x: x[3], reverse=True)[0]
                    if the_best_result:
                        search_result["BestID"] = the_best_result.ID
                        search_result["BestEtalon"] = the_best_result.Etalon
                        search_result["BestScore"] = the_best_result.Score
                        answers_search_result = await self.es.answer_search(self.prm.answers_index, the_best_result.ID,
                                                                            pubid)
                        if answers_search_result["search_results"]:
                            search_result["BestAnswer"] = answers_search_result["search_results"][0]["templateText"]
                        search_result["BestElasticNum"] = candidates.index(the_best_result)
                    else:
                        search_result["BestID"] = 0
                        search_result["BestEtalon"] = "NO"
                        search_result["BestScore"] = "NO"
                        search_result["BestAnswer"] = "NO"
                        search_result["BestElasticNum"] = 0
                    return search_result
                else:
                    logger.info("elasticsearch doesn't find any etalons for input text {}".format(str(text)))
                    return {"templateId": 0, "templateText": ""}
            else:
                logger.info("tokenizer returned empty value for input text {}".format(str(text)))
                return {"templateId": 0, "templateText": ""}
                # return tuple(0, "no", "no", 0)
        except Exception:
            logger.exception("Searching problem with text: {}".format(str(text)))
            # return tuple(0, "no", "no", 0)
            return {"templateId": 0, "templateText": ""}
