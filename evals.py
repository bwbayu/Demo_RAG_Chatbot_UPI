import json
from search import classify_query, search_dense_index, search_sparse_index, rrf_fusion, reranking_results, context_generation
from rouge import Rouge

def retrieval_pipeline(query, top_k=10):
    # eval retrieval
    classified_type = classify_query(query, chat_history="")
    dense_results = search_dense_index(query, filter_types=classified_type)
    sparse_results = search_sparse_index(query, filter_types=classified_type)
    fused_results = rrf_fusion(dense_results, sparse_results, top_n=top_k)
    fused_docs = [result['text'] for result in fused_results]
    rerank_results = reranking_results(query, fused_docs, fused_results)
    top_ids = [data['id'] for data in rerank_results]
    return top_ids, rerank_results

def evaluate_rag(eval_data, k=10, evaluating_generation=True, save_path=None):
    correct = 0
    total = 0
    reciprocal_ranks = []
    scores = []
    rouge = Rouge()

    for item in eval_data:
        # retrieval evaluation
        query, gold_ids, gold_answer = (
            item["query"],
            set(item["gold_doc_ids"]),
            item["gold_answer"],
        )
        retrieved, reranking_results = retrieval_pipeline(query, top_k=k)

        if gold_ids.intersection(retrieved):
            correct += 1

        for rank, doc in enumerate(retrieved, start=1):
            if doc in gold_ids:
                reciprocal_ranks.append(1.0 / rank)
                break

        total += 1

        # generation evaluation
        if evaluating_generation:
            generation_output = context_generation(
                query, contexts=reranking_results, chat_history="", streaming=False
            )
            item["generated_answer"] = generation_output
            if generation_output.strip():
                score = rouge.get_scores(generation_output, gold_answer, avg=True)
                scores.append(score["rouge-l"]["f"])

    # metrics
    recall_at_k = correct / total
    mrr = sum(reciprocal_ranks) / len(eval_data)

    if evaluating_generation:
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(eval_data, f, ensure_ascii=False, indent=2)
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return {"recall@k": recall_at_k, "MRR": mrr, "Avg ROUGE-L": avg_score}

    return {"recall@k": recall_at_k, "MRR": mrr}


if __name__ == "__main__":
    with open("data/eval/rag_eval.json", "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    retrieval_result = evaluate_rag(
        eval_data=eval_data,
        k=10,
        evaluating_generation=False,
        save_path="data/eval/rag_eval_with_gen.json",
    )

    print(retrieval_result)
