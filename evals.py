import json
from collections import defaultdict
from rouge import Rouge
import math
from search import classify_query, search_dense_index, search_sparse_index, rrf_fusion, reranking_results, context_generation

def retrieval_pipeline(query, top_k=10):
    # # classify docs
    classified_type = classify_query(query, chat_history="")
    dense_results, dense_results2 = search_dense_index(query, filter_types=classified_type)
    sparse_results, sparse_results2 = search_sparse_index(query, filter_types=classified_type)
    fused_results = rrf_fusion(dense_results, sparse_results, top_n=top_k)
    fused_docs = [r['text'] for r in fused_results]
    # non-classify docs
    fused_results2 = rrf_fusion(dense_results2, sparse_results2, top_n=top_k)
    fused_docs2 = [r['text'] for r in fused_results2]
    # combine from two docs
    fused_results.extend(fused_results2)
    fused_docs.extend(fused_docs2)
    rerank = reranking_results(query, fused_docs, fused_results, top_k)
    top_ids = [d['id'] for d in rerank]
    return top_ids, rerank

def precision_at_k(retrieved, gold_ids, k):
    # get top k document from retrieved document
    retrieved_k = retrieved[:k]
    # count all relevant document based on gold_ids
    rel = sum(1 for doc_id in retrieved_k if doc_id in gold_ids)
    # calcualte precision
    return rel / k if k else 0.0

def average_precision(retrieved, gold_ids, k):
    ap = 0.0
    rel_count = 0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        # check if that doc_id is appear in gold_ids
        if doc_id in gold_ids:
            # count number of relevant document
            rel_count += 1
            # sum of all precision
            ap += rel_count / i
    # calculate mAP
    return ap / len(gold_ids) if gold_ids else 0.0

def ndcg_at_k(retrieved, gold_ids, k):
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        # check if that doc_id is appear in gold_ids
        if doc_id in gold_ids:
            # calculate dcg (Discounted Cumulative Gain)
            dcg += 1.0 / math.log2(i + 1)
    # Ideal DCG (all relevant at the top)
    ideal_hits = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_rag(eval_data, k=10, evaluating_generation=True, save_path=None):
    rouge = Rouge()

    total = 0
    overall_correct = 0
    overall_rr_sum = 0.0
    overall_p_sum = 0.0   # Precision@k
    overall_ap_sum = 0.0  # MAP
    overall_ndcg_sum = 0.0  # NDCG

    type_total = defaultdict(int)
    type_correct = defaultdict(int)
    type_rr_sum = defaultdict(float)
    type_p_sum = defaultdict(float)  
    type_ap_sum = defaultdict(float) 
    type_ndcg_sum = defaultdict(float) 

    gen_scores = []

    for item in eval_data:
        query = item["query"]
        gold_ids = set(item["gold_doc_ids"])
        gold_answer = item.get("gold_answer", "")
        item_type = item.get("type", "other")

        retrieved, rerank_ctx = retrieval_pipeline(query, top_k=k)
        item['retrieve_ids'] = retrieved

        # --- Metrics ---
        hit_rank = None
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in gold_ids:
                hit_rank = rank
                break

        total += 1
        type_total[item_type] += 1

        if hit_rank is not None:
            overall_correct += 1
            type_correct[item_type] += 1
            rr = 1.0 / hit_rank
            overall_rr_sum += rr
            type_rr_sum[item_type] += rr

        # Precision@k
        p_at_k = precision_at_k(retrieved, gold_ids, k)
        overall_p_sum += p_at_k
        type_p_sum[item_type] += p_at_k

        # MAP
        ap = average_precision(retrieved, gold_ids, k)
        overall_ap_sum += ap
        type_ap_sum[item_type] += ap

        # NDCG
        ndcg = ndcg_at_k(retrieved, gold_ids, k)
        overall_ndcg_sum += ndcg
        type_ndcg_sum[item_type] += ndcg

        # --- Generation evaluation ---
        if evaluating_generation:
            generation_output = context_generation(
                query, contexts=rerank_ctx, chat_history="", streaming=False
            )
            item["generated_answer"] = generation_output
            if generation_output and gold_answer:
                score = rouge.get_scores(generation_output, gold_answer, avg=True)
                gen_scores.append(score["rouge-l"]["f"])

    overall = {
        "count": total,
        "recall@k": (overall_correct / total) if total else 0.0,
        "MRR": (overall_rr_sum / total) if total else 0.0,
        "Precision@k": (overall_p_sum / total) if total else 0.0,
        "MAP": (overall_ap_sum / total) if total else 0.0,       
        "NDCG@k": (overall_ndcg_sum / total) if total else 0.0,  
    }

    by_type = {}
    for t in sorted(type_total.keys()):
        n = type_total[t]
        by_type[t] = {
            "count": n,
            "recall@k": (type_correct[t] / n) if n else 0.0,
            "MRR": (type_rr_sum[t] / n) if n else 0.0,
            "Precision@k": (type_p_sum[t] / n) if n else 0.0,
            "MAP": (type_ap_sum[t] / n) if n else 0.0,       
            "NDCG@k": (type_ndcg_sum[t] / n) if n else 0.0,  
        }

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)

    result = {"overall": overall, "by_type": by_type}
    if evaluating_generation:
        result["Avg ROUGE-L"] = (sum(gen_scores) / len(gen_scores)) if gen_scores else 0.0

    return result


if __name__ == "__main__":
    with open("data/eval/rag_eval.json", "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    results = evaluate_rag(
        eval_data=eval_data,
        k=10,
        evaluating_generation=False,
        save_path="data/eval/rag_eval_combine.json"
    )

    print("=== OVERALL ===")
    print(results["overall"])
    print("\n=== BY TYPE ===")
    for t, m in results["by_type"].items():
        print(f"{t:>20}: {m}")
