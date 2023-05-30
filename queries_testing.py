import os
import pandas as pd
import requests

queies_df = pd.read_csv(os.path.join("data", "search_queries12538.csv"), sep="\t")
print(queies_df)
test_results = []
for num, q in enumerate(list(queies_df["Query"])):
    print(num, q)
    q_request = {"pubid": 9, "text": q}
    res = requests.post("http://0.0.0.0:8080/api/search", json=q_request)
    res_dict = res.json()
    res_dict["Query"] = q
    test_results.append(res_dict)

test_results_df = pd.DataFrame(test_results)
print(test_results_df)
test_results_df.to_csv(os.path.join("data", "results", "fa_test_results12538.csv"), sep="\t", index=False)