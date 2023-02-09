import sys
import json
import argparse
import drqa_retriever as retriever


def main(args):

    with open(args.qa_file) as f:
        qa_file = json.load(f)
    questions = []
    for ex in qa_file:
        if ex['scenario'] != "":
            if ex['scenario'][-1] != '.':
                questions.append(ex['scenario'] + ". " + ex['question'])
            else:
                questions.append(ex['scenario'] + " " + ex['question'])
        else:
            questions.append(ex['question'])

    top_ids_and_scores = []
    for question in questions:
        psg_ids, scores = ranker.closest_docs(question, args.n_docs)
        top_ids_and_scores.append((psg_ids, scores.tolist()))

    with open(args.db_path) as f:
        id2snippet = json.load(f)

    # validate
    matches = []
    for ex, top_psg_ids in zip(qa_file, top_ids_and_scores):
        matches.append([id2snippet[curr_id] == ex['snippet'] for curr_id in top_psg_ids[0]])

    print(args.qa_file)
    for top_n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 25, 100]:
        count = sum([any(curr_match[:top_n]) for curr_match in matches])
        print("Top {}: {:.1f}".format(top_n, count / len(matches) * 100))

    with open(args.out_file, 'w') as f:
        json.dump(top_ids_and_scores, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa_file', required=True, type=str, default=None)
    parser.add_argument('--dpr_path', type=str, default="../DPR")
    parser.add_argument('--db_path', type=str, default=None)
    parser.add_argument('--tfidf_path', type=str, default=None)
    parser.add_argument('--out_file', type=str, default=None)
    parser.add_argument('--match', type=str, default='string', choices=['regex', 'string'])
    parser.add_argument('--n-docs', type=int, default=100)
    parser.add_argument('--validation_workers', type=int, default=16)
    args = parser.parse_args()

    sys.path.append(args.dpr_path)

    ranker = retriever.get_class('tfidf')(tfidf_path=args.tfidf_path)

    main(args)


